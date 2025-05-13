from typing import Any, List, Dict
from ray.util.client.common import ClientObjectRef
from pyjava.api.mlsql import RayContext
from pyjava.storage import streaming_tar
import os
import inspect
import functools
import json
from typing import Type, Any

from typing import Dict, Generator, Optional
from dataclasses import dataclass
from byzerllm.utils import (
    print_flush,
    format_prompt_jinja2,
    format_str_jinja2,
)
from .store import transfer_from_ob
from byzerllm.utils.types import SingleOutputMeta
from pydantic import BaseModel


@dataclass
class BlockRow:
    start: int
    offset: int
    value: bytes


def restore_model(conf: Dict[str, str], target_dir: str):
    model_servers = RayContext.parse_servers(conf["modelServers"])
    model_binary = RayContext.collect_from(model_servers)
    streaming_tar.save_rows_as_file(model_binary, target_dir)


def load_model(target_dir: str) -> Generator[BlockRow, None, None]:
    model_binary = streaming_tar.build_rows_from_file(target_dir)
    return model_binary


def consume_model(conf: Dict[str, str]):
    # consume the model server to prevent socket server leak.
    # hoverer,  model server may have be consumed by other worker
    # so just try to consume it
    try:
        model_servers = RayContext.parse_servers(conf["modelServers"])
        for item in RayContext.collect_from(model_servers):
            pass
    except Exception as e:
        pass


def common_init_model(
    model_refs: List[ClientObjectRef],
    conf: Dict[str, str],
    model_dir: str,
    is_load_from_local: bool,
):

    udf_name = conf["UDF_CLIENT"] if "UDF_CLIENT" in conf else "UNKNOW MODEL"

    if not is_load_from_local:
        if "standalone" in conf and conf["standalone"] == "true":
            print_flush(
                f"MODEL[{udf_name}] Standalone mode: restore model to {model_dir} directly from model server"
            )
            restore_model(conf, model_dir)
        else:
            print_flush(
                f"MODEL[{udf_name}] Normal mode: restore model from ray object store to {model_dir}"
            )
            if not os.path.exists(model_dir):
                transfer_from_ob(udf_name, model_refs, model_dir)
    else:
        print_flush(
            f"MODEL[{udf_name}]  Local mode: Load model from local path ({model_dir}), consume the model server to prevent socket server leak."
        )
        consume_model(conf)


def parse_params(params: Dict[str, str], prefix: str):
    import json

    new_params = {}
    for k, v in params.items():
        if k.startswith(f"{prefix}."):
            # sft.float.num_train_epochs
            tpe = k.split(".")[1]
            new_k = k.split(".")[2]
            new_v = v
            if tpe == "float":
                new_v = float(v)
            elif tpe == "int":
                new_v = int(v)
            elif tpe == "bool":
                new_v = v == "true"
            elif tpe == "str":
                new_v = v
            elif tpe == "list":
                new_v = json.loads(v)
            elif tpe == "dict":
                new_v = json.loads(v)
            new_params[new_k] = new_v
    return new_params


import inspect


def check_param_exists(func, name):
    return name in inspect.signature(func).parameters


# add a log funcition to log the string to a specified file
def log_to_file(msg: str, file_path: str):
    with open(file_path, "a") as f:
        f.write(msg)
        f.write("\n")


class _PromptWraper:
    def __init__(
        self, func, llm, render, check_result, options, *args, **kwargs
    ) -> None:
        self.func = func
        self.llm = llm
        self.render = render
        self.check_result = check_result
        self.args = args
        self.kwargs = kwargs
        self._options = options

    def options(self, options: Dict[str, Any]):
        self._options = {**self._options, **options}
        return self

    def with_llm(self, llm):
        self.llm = llm
        return self

    def prompt(self):
        func = self.func
        render = self.render
        args = self.args
        kwargs = self.kwargs

        signature = inspect.signature(func)
        arguments = signature.bind(*args, **kwargs)
        arguments.apply_defaults()
        input_dict = {}
        for param in signature.parameters:
            input_dict.update({param: arguments.arguments[param]})

        new_input_dic = func(**input_dict)
        if new_input_dic and not isinstance(new_input_dic, dict):
            raise TypeError(f"Return value of {func.__name__} should be a dict")
        if new_input_dic:
            input_dict = {**input_dict, **new_input_dic}

        return format_prompt_jinja2(func, **input_dict)
    
    def run(self):
        func = self.func
        llm = self.llm
        render = self.render
        check_result = self.check_result
        args = self.args
        kwargs = self.kwargs

        signature = inspect.signature(func)
        arguments = signature.bind(*args, **kwargs)
        arguments.apply_defaults()
        input_dict = {}
        for param in signature.parameters:
            input_dict.update({param: arguments.arguments[param]})

        is_lambda = inspect.isfunction(llm) and llm.__name__ == "<lambda>"
        if is_lambda:
            if "self" in input_dict:
                instance = input_dict.pop("self")
                return llm(instance).prompt(
                    render=render, check_result=check_result, options=self._options
                )(func)(instance, **input_dict)

        if isinstance(llm, ByzerLLM) or isinstance(llm, SimpleByzerLLM):
            if "self" in input_dict:
                instance = input_dict.pop("self")
                return llm.prompt(
                    render=render, check_result=check_result, options=self._options
                )(func)(instance, **input_dict)
            else:
                return llm.prompt(
                    render=render, check_result=check_result, options=self._options
                )(func)(**input_dict)

        if isinstance(llm, str):
            _llm = ByzerLLM()
            _llm.setup_default_model_name(llm)
            _llm.setup_template(llm, "auto")

            if "self" in input_dict:
                instance = input_dict.pop("self")
                return _llm.prompt(
                    render=render, check_result=check_result, options=self._options
                )(func)(instance, **input_dict)
            else:
                return _llm.prompt(
                    render=render, check_result=check_result, options=self._options
                )(func)(**input_dict)

        raise ValueError(
            "llm should be a lambda function or ByzerLLM instance or a string of model name"
        )


def prompt_lazy(
    llm=None,
    render: str = "jinja2",
    check_result: bool = False,
    options: Dict[str, Any] = {},
):
    def _impl(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            pw = _PromptWraper(
                func, llm, render, check_result, options, *args, **kwargs
            )
            return pw

        return wrapper

    return _impl


class MetaOutput(BaseModel):
    input_tokens_count: int
    generated_tokens_count: int
    reasoning_content: str
    finish_reason: str
    first_token_time: float
class MetaHolder:
    def __init__(self, meta: Optional[Any] = None) -> None:
        self.meta = meta  

    def get_meta(self):
        if isinstance(self.meta, dict):
            return self.meta
        if isinstance(self.meta, SingleOutputMeta):
            return {
                "input_tokens_count": self.meta.input_tokens_count,
                "generated_tokens_count": self.meta.generated_tokens_count,
                "reasoning_content": self.meta.reasoning_content,
                "finish_reason": self.meta.finish_reason,
                "first_token_time": self.meta.first_token_time
            }
        return self.meta
    
    def get_meta_model(self):
        v = self.get_meta()
        
        if v is None:
            return None
        
        return MetaOutput(
            input_tokens_count=v.get("input_tokens_count", 0),
            generated_tokens_count=v.get("generated_tokens_count", 0),
            reasoning_content=v.get("reasoning_content", ""),
            finish_reason=v.get("finish_reason", ""),
            first_token_time=v.get("first_token_time", 0.0)            
        )

class _PrompRunner:
    def __init__(
        self,
        func,
        instance,
        llm,
        render: str,
        check_result: bool,
        options: Dict[str, Any],
    ) -> None:
        self.func = func
        self.instance = instance
        self.llm = llm
        self.render = render
        self.check_result = check_result
        self._options = options
        self.max_turns = 10
        self.response_markers = None
        self.auto_remove_response_markers_from_output = True
        self.extractor = None
        self.continue_prompt = "接着前面的内容继续"
        self.response_markers_template = """你的输出可能会被切分成多轮对话完成。请确保第一次输出以{{ RESPONSE_START }}开始，输出完毕后，请使用{{ RESPONSE_END }}标记。中间的回复不需要使用这两个标记。"""
        self.model_class = None
        self.return_prefix = None
        self.stop_suffix_list = None
        self.meta_holder = MetaHolder(None)
        self.conversation = []

    def with_meta(self, meta_holder):
        self.meta_holder = meta_holder
        return self

    def with_conversation(self, conversation: List[Dict[str,Any]]):
        self.conversation = conversation
        return self    

    def with_return_type(self, model_class: Type[Any]):
        self.model_class = model_class
        return self
    
    def with_stop_suffix_list(self, suffix_list: List[str]):
        self.stop_suffix_list = suffix_list
        if "llm_config" in self._options:
            self._options["llm_config"]["gen.stop"] = suffix_list
        else:
            self._options["llm_config"] = {"gen.stop": suffix_list}
        return self

    def with_continue_prompt(self, prompt: str):
        self.continue_prompt = prompt
        return self      

    def with_return_prefix(self, prefix: str):
        self.return_prefix = prefix
        return self

    def __call__(self, *args, **kwargs) -> Any:
        if self.llm:
            return self.run(*args, **kwargs)
        return self.prompt(*args, **kwargs)

    def options(self, options: Dict[str, Any]):
        self._options = {**self._options, **options}
        return self

    def with_response_markers(
        self,
        response_markers: Optional[List[str]] = None,
        response_markers_template: Optional[str] = None,
    ):
        if response_markers is not None and len(response_markers) != 2:
            raise ValueError("response_markers should be a list of two elements")
        if response_markers is None:
            response_markers = ["<RESPONSE>", "</RESPONSE>"]
        self.response_markers = response_markers
        if response_markers_template:
            self.response_markers_template = response_markers_template
        return self

    def with_extractor(self, func):
        self.extractor = func
        return self

    def with_max_turns(self, max_turns: int):
        self.max_turns = max_turns
        return self

    def prompt(self, *args, **kwargs):
        signature = inspect.signature(self.func)
        if self.instance:
            arguments = signature.bind(self.instance, *args, **kwargs)
        else:
            arguments = signature.bind(*args, **kwargs)

        arguments.apply_defaults()
        input_dict = {}
        for param in signature.parameters:
            input_dict.update({param: arguments.arguments[param]})

        new_input_dic = self.func(**input_dict)
        if new_input_dic and not isinstance(new_input_dic, dict):
            raise TypeError(f"Return value of {self.func.__name__} should be a dict")
        if new_input_dic:
            input_dict = {**input_dict, **new_input_dic}

        if "self" in input_dict:
            input_dict.pop("self")
        
        return format_prompt_jinja2(self.func, **input_dict)        

    def with_llm(self, llm):
        self.llm = llm
        return self

    def with_auto_remove_response_markers(self, flag: bool):
        self.auto_remove_response_markers_from_output = flag
        return self

    def _remove_response_markers(self, output: str):
        [start, end] = self.response_markers
        start_index = output.find(start)
        end_index = output.find(end)
        return output[start_index + len(start) : end_index]

    def _multi_turn_wrapper_with_generator(
        self,
        llm,
        v: List[Any],
        signature: inspect.Signature,
        origin_input: Optional[str] = None,
    ):
        conversations = []
        s = ""
        for item in v:
            s += item[0]
            yield item
        if origin_input is None:
            raise ValueError(
                "origin_input should be set when return value is a generator"
            )

        conversations = []
        conversations.append({"role": "user", "content": origin_input})
        conversations.append({"role": "assistant", "content": s})
        turn = 1
        end_marker = self.response_markers[1]
        while turn < self.max_turns and end_marker not in s:
            conversations.append({"role": "user", "content": self.continue_prompt})
            temp_options = {**{"delta_mode": True}, **self._options}
            v1 = llm.stream_chat_oai(conversations=conversations, **temp_options)
            temp = ""
            for item in v1:
                temp += item[0]
                yield item
            conversations.append({"role": "assistant", "content": temp})
            s += temp
            turn += 1
        return

    def _multi_turn_wrapper(
        self,
        llm,
        v: List[Any],
        signature: inspect.Signature,
        origin_input: Optional[str] = None,
    ):

        if not issubclass(signature.return_annotation, str):
            raise ValueError(
                "Return value of function should be a string when response_markers is set"
            )

        conversations = []
        response = v[0]
        s = response.output
        conversations += response.input["history"] or []
        conversations.append({"role": "user", "content": response.input["instruction"]})
        conversations.append({"role": "assistant", "content": response.output})
        turn = 1
        end_marker = self.response_markers[1]

        while turn < self.max_turns and end_marker not in response.output:
            conversations.append({"role": "user", "content": self.continue_prompt})
            v1 = llm.chat_oai(conversations=conversations, **self._options)
            response = v1[0]
            conversations.append({"role": "assistant", "content": response.output})
            s += response.output
            turn += 1

        if self.auto_remove_response_markers_from_output:
            output_content = self._remove_response_markers(output=s)
        response.output = output_content
        if self.extractor:
            if isinstance(self.extractor, str):
                return self.extractor(response)
            else:
                return self.extractor(response.output)
        return response.output

    def is_instance_of_generator(self, v):
        from typing import Generator, get_origin, get_args
        import collections

        if get_origin(v) is collections.abc.Generator:
            args = get_args(v)
            if args == (str, type(None), type(None)):
                return True
        return False

    def to_model(self,result: str):
        from byzerllm.utils import str2model
        return str2model.to_model(result, self.model_class)

    def run(self, *args, **kwargs):
        func = self.func
        llm = self.llm
        render = self.render
        check_result = self.check_result

        signature = inspect.signature(func)
        if self.instance:
            arguments = signature.bind(self.instance, *args, **kwargs)
        else:
            arguments = signature.bind(*args, **kwargs)

        arguments.apply_defaults()
        input_dict = {}
        for param in signature.parameters:
            input_dict.update({param: arguments.arguments[param]})

        is_lambda = inspect.isfunction(llm) and llm.__name__ == "<lambda>"
        if is_lambda:
            return_origin_response = True if self.response_markers else False
            marker = None

            if self.response_markers:
                marker = format_str_jinja2(
                    self.response_markers_template,
                    RESPONSE_START=self.response_markers[0],
                    RESPONSE_END=self.response_markers[1],
                )

            origin_input = self.prompt(*args, **kwargs)
            if self.response_markers:
                origin_input = f"{origin_input}\n\n{marker}"

            v = llm(self.instance).prompt(
                render=render,
                check_result=check_result,
                options=self._options,
                return_origin_response=return_origin_response,
                marker=marker,
                assistant_prefix=self.return_prefix,
                meta_holder=self.meta_holder,
                conversation=self.conversation,
            )(func)(**input_dict)
            prefix = self.return_prefix if self.return_prefix else ""
            if not return_origin_response:                
                if self.extractor:
                    v = self.extractor(f"{prefix}{v}")
                if self.model_class:
                    return self.to_model(f"{prefix}{v}")
                return v

            if self.is_instance_of_generator(signature.return_annotation):
                return self._multi_turn_wrapper_with_generator(
                    llm(self.instance), v, signature, origin_input=origin_input
                )

            v = self._multi_turn_wrapper(
                llm(self.instance), v, signature, origin_input=origin_input
            )
            if self.model_class:
                return self.to_model(f"{prefix}{v}")
            return v

        if isinstance(llm, ByzerLLM) or isinstance(llm, SimpleByzerLLM):
            return_origin_response = True if self.response_markers else False
            marker = None
            if self.response_markers:
                marker = format_str_jinja2(
                    self.response_markers_template,
                    RESPONSE_START=self.response_markers[0],
                    RESPONSE_END=self.response_markers[1],
                )

            origin_input = self.prompt(*args, **kwargs)
            if self.response_markers:
                origin_input = f"{origin_input}\n\n{marker}"

            v = llm.prompt(
                render=render,
                check_result=check_result,
                options=self._options,
                return_origin_response=return_origin_response,
                marker=marker,
                assistant_prefix=self.return_prefix,
                meta_holder=self.meta_holder,
                conversation=self.conversation,
            )(func)(**input_dict)
            prefix = self.return_prefix if self.return_prefix else ""
            if not return_origin_response:                
                if self.extractor:
                    v = self.extractor(f"{prefix}{v}")
                if self.model_class:
                    return self.to_model(f"{prefix}{v}")
                return v

            if self.is_instance_of_generator(signature.return_annotation):
                return self._multi_turn_wrapper_with_generator(
                    llm, v, signature, origin_input=origin_input
                )

            v = self._multi_turn_wrapper(llm, v, signature, origin_input=origin_input)            
            return v

        if isinstance(llm, str):
            _llm = ByzerLLM()
            _llm.setup_default_model_name(llm)
            _llm.setup_template(llm, "auto")
            return_origin_response = True if self.response_markers else False
            marker = None

            if self.response_markers:
                marker = format_str_jinja2(
                    self.response_markers_template,
                    RESPONSE_START=self.response_markers[0],
                    RESPONSE_END=self.response_markers[1],
                )

            origin_input = self.prompt(*args, **kwargs)
            if self.response_markers:
                origin_input = f"{origin_input}\n\n{marker}"

            v = _llm.prompt(
                render=render,
                check_result=check_result,
                options=self._options,
                return_origin_response=return_origin_response,
                marker=marker,
                assistant_prefix=self.return_prefix,
                meta_holder=self.meta_holder,
                conversation=self.conversation,
            )(func)(**input_dict)
            prefix = self.return_prefix if self.return_prefix else ""
            if not return_origin_response:
                if self.extractor:
                    v = self.extractor(f"{prefix}{v}")
                if self.model_class:
                    return self.to_model(self.model_class)(lambda: f"{prefix}{v}")()
                return v

            if self.is_instance_of_generator(signature.return_annotation):
                return self._multi_turn_wrapper_with_generator(
                    llm, v, signature, origin_input=origin_input
                )

            v = self._multi_turn_wrapper(llm, v, signature, origin_input=origin_input)
            if self.model_class:
                return self.to_model(self.model_class)(lambda: f"{prefix}{v}")()
            return v

        else:
            raise ValueError(
                "llm should be a lambda function or ByzerLLM instance or a string of model name"
            )


class _DescriptorPrompt:
    def __init__(
        self,
        func,
        wrapper,
        llm,
        render: str,
        check_result: bool,
        options: Dict[str, Any],
    ):
        self.func = func
        self.wrapper = wrapper
        self.llm = llm
        self.render = render
        self.check_result = check_result
        self._options = options
        self.prompt_runner = _PrompRunner(
            self.wrapper,
            None,
            self.llm,
            self.render,
            self.check_result,
            options=self._options,
        )

    def __get__(self, instance, owner):
        if instance is None:
            return self
        else:
            return _PrompRunner(
                self.wrapper,
                instance,
                self.llm,
                self.render,
                self.check_result,
                options=self._options,
            )

    def reset(self):
        self.prompt_runner = _PrompRunner(
            self.wrapper,
            None,
            self.llm,
            self.render,
            self.check_result,
            options=self._options,
        )
        return self

    def with_response_markers(
        self,
        response_markers: Optional[List[str]] = None,
        response_markers_template: Optional[str] = None,
    ):
        self.prompt_runner.with_response_markers(
            response_markers=response_markers,
            response_markers_template=response_markers_template,
        )
        return self

    def with_auto_remove_response_markers(self, flag: bool):
        self.prompt_runner.with_auto_remove_response_markers(flag)
        return self

    def with_max_turns(self, max_turns: int):
        self.prompt_runner.with_max_turns(max_turns)
        return self

    def with_continue_prompt(self, prompt: str):
        self.prompt_runner.with_continue_prompt(prompt)
        return self
    
    def with_stop_suffix_list(self, suffix_list: List[str]):
        self.prompt_runner.with_stop_suffix_list(suffix_list)
        return self

    def with_return_type(self, model_class: Type[Any]):
        self.prompt_runner.with_return_type(model_class)
        return self
    
    def with_return_prefix(self, prefix: str):
        self.prompt_runner.with_return_prefix(prefix)
        return self
    
    def with_meta(self, meta_holder):
        self.prompt_runner.with_meta(meta_holder)
        return self

    def with_conversation(self, conversation: List[Dict[str,Any]]):
        self.prompt_runner.with_conversation(conversation)
        return self    

    def __call__(self, *args, **kwargs):
        return self.prompt_runner(*args, **kwargs)

    def prompt(self, *args, **kwargs):
        return self.prompt_runner.prompt(*args, **kwargs)

    def run(self, *args, **kwargs):
        return self.prompt_runner.run(*args, **kwargs)

    def with_llm(self, llm):
        self.llm = llm
        self.prompt_runner.with_llm(llm)
        return self

    def with_extractor(self, func):
        self.prompt_runner.with_extractor(func)
        return self

    def options(self, options: Dict[str, Any]):
        self._options = {**self._options, **options}
        self.prompt_runner.options(options)
        return self


class prompt:
    def __init__(
        self,
        llm=None,
        render: str = "jinja2",
        check_result: bool = False,
        options: Dict[str, Any] = {},
    ):
        self.llm = llm
        self.render = render
        self.check_result = check_result
        self.options = options

    def __call__(self, func):        
        wrapper = func
        return self._make_wrapper(func, wrapper)

    def _make_wrapper(self, func, wrapper):
        return _DescriptorPrompt(
            func,
            wrapper,
            self.llm,
            self.render,
            self.check_result,
            options=self.options,
        )


from byzerllm.utils.client import ByzerLLM,SimpleByzerLLM
from byzerllm.utils.retrieval import ByzerRetrieval
from byzerllm.utils.connect_ray import connect_cluster
from byzerllm.apps.agent.registry import reply as agent_reply
from byzerllm.utils.nontext import Image
from .utils.llms import get_model_info, get_single_llm, get_llm_names

__all__ = [
    "ByzerLLM",
    "SimpleByzerLLM",
    "ByzerRetrieval",
    "connect_cluster",
    "prompt",
    "agent_reply",
    "Image",
    "get_model_info",
    "get_single_llm",
    "get_llm_names",
]
