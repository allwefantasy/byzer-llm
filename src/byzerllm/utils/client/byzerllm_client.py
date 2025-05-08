from pyjava import PythonContext, RayContext
from typing import Dict, Any, List, Optional, Union, Tuple, Callable, Annotated
from pyjava.udf import UDFBuilder
import ray
from ray.util.client.common import ClientActorHandle, ClientObjectRef
from byzerllm.utils.client import code_utils
from byzerllm.utils import (
    function_calling_format,
    response_class_format,
    response_class_format_after_chat,
    FunctionCallList,
    function_impl_format,
    base_ability_format,
    BaseAbility,
    sys_response_class_format,
    sys_function_calling_format,
    sys_function_impl_format,
    exec_capture_output,    
    format_prompt_jinja2,
)
from byzerllm.utils.ray_utils import cancel_placement_group, get_actor_info
from byzerllm.utils.json_repaire import repair_json_str
import byzerllm
import json
import importlib
import time
import functools
import inspect
import pydantic
import copy
import traceback
from enum import Enum
from loguru import logger
import asyncio

from byzerllm.utils.client.types import (
    Templates,
    Template,
    Role,
    LLMHistoryItem,
    LLMRequest,
    LLMFunctionCallResponse,
    LLMClassResponse,
    InferBackend,
    EventName,
    EventCallbackResult,
    EventCallback,
    LLMResponse,
    FintuneRequestExtra,
    FintuneRequest,
    ExecuteCodeResponse,
    LLMMetadata,
)


class ByzerLLM:

    def __init__(self, url: Optional[str] = None, **kwargs):
        self.url = url
        self.default_sys_conf = {
            "pythonMode": "ray",
            "maxConcurrency": 1,
            "num_gpus": 1,
            "masterMaxConcurrency": 1000,
            "workerMaxConcurrency": 1,
            "infer_backend": "transformers",
        }
        self.sys_conf = self.default_sys_conf.copy()
        self.sql_model = "context" in globals()

        self.verbose = kwargs.get("verbose", False)

        self.force_skip_context_length_check = False
        if "force_skip_context_length_check" in kwargs:
            self.force_skip_context_length_check = kwargs[
                "force_skip_context_length_check"
            ]

        self.mapping_auto_use_apply_chat_template = {}

        self.mapping_max_input_length = {}
        self.mapping_max_output_length = {}
        self.mapping_max_model_length = {}
        self.mapping_role_mapping = {}
        self.mapping_extra_generation_params = {}
        self.mapping_clean_func = {}

        self.mapping_function_calling_format_func = {}
        self.mapping_response_class_format_func = {}
        self.mapping_response_class_format_after_chat_func = {}
        self.mapping_impl_func_format_func = {}

        self.mapping_base_system_message = {}
        self.mapping_sys_response_class_format_func = {}
        self.mapping_sys_function_calling_format_func = {}
        self.mapping_sys_response_class_format_after_chat_func = {}
        self.mapping_sys_impl_func_format_func = {}

        self.func_impl_cache = {}
        self.meta_cache = {}

        self.byzer_engine_url = None
        if "byzer_engine_url" in kwargs:
            self.byzer_engine_url = kwargs["byzer_engine_url"]

        self.default_max_output_length = 4000
        if "default_max_output_length" in kwargs:
            self.default_max_output_length = kwargs["default_max_output_length"]

        self.default_model_name = None
        self.default_emb_model_name = None
        self.default_rerank_model_name = None
        self.default_role_mapping = {
            "user_role": "User:",
            "assistant_role": "Assistant:",
            "system_msg": "You are a helpful assistant. Think it over and answer the user question correctly.",
        }

        self.pin_model_worker_mapping = None

        if url is not None and self.sql_model:
            v = globals()
            self.context = v["context"]
            self.ray_context = RayContext.connect(v, self.url, **kwargs)
        else:
            self.context = PythonContext(0, [], self.sys_conf)
            self.context.have_fetched = True
            self.ray_context = self.context.rayContext

        self.event_callbacks: Dict[EventName, List[EventCallback]] = {}
        self.sub_clients = {}
        self.skip_nontext_check = False

    @property
    def metadata(self) -> LLMMetadata:
        meta = self.get_meta(model=self.default_model_name)
        return LLMMetadata(
            context_window=meta.get("max_model_len", 8192),
            num_output=meta.get("num_output", 256),
            is_chat_model=not meta.get("embedding_mode", False),
            is_function_calling_model=True,
            model_name=meta.get("model_name", self.default_model_name),
        )

    @staticmethod
    def from_default_model(model: str, auto_connect_cluster: bool = True) -> "ByzerLLM":
        if auto_connect_cluster:
            byzerllm.connect_cluster()
        llm = ByzerLLM()
        llm.setup_default_model_name(model)
        return llm

    def setup_sub_client(
        self, client_name: str, client: Union[List["ByzerLLM"], "ByzerLLM"] = None
    ) -> "ByzerLLM":
        if isinstance(client, list):
            self.sub_clients[client_name] = client
        else:
            self.sub_clients[client_name] = client
        return self

    def get_sub_client(self, client_name: str) -> Union[List["ByzerLLM"], Optional["ByzerLLM"]]:
        return self.sub_clients.get(client_name, None)

    def remove_sub_client(self, client_name: str) -> "ByzerLLM":
        if client_name in self.sub_clients:
            del self.sub_clients[client_name]
        return self

    def add_event_callback(
        self, event_name: EventName, callback: EventCallback
    ) -> None:
        self.event_callbacks.setdefault(event_name, []).append(callback)

    def _trigger_event(self, event_name: EventName, *args, **kwargs) -> Optional[Any]:
        if event_name in self.event_callbacks:
            for callback in self.event_callbacks[event_name]:
                continue_flag, value = callback(*args, **kwargs)
                if not continue_flag:
                    return value
        return None

    def setup_reset(self):
        self.sys_conf = self.default_sys_conf.copy()
        self.context.conf = self.sys_conf

    def setup_pin_model_worker_mapping(
        self, pin_model_worker_mapping: Dict[Any, int]
    ) -> "ByzerLLM":
        self.pin_model_worker_mapping = pin_model_worker_mapping
        return self

    def setup_load_balance_way(self, load_balance_way: str) -> "ByzerLLM":
        self.sys_conf["load_balance"] = load_balance_way
        return self

    def setup_default_model_name(self, model_name: str) -> "ByzerLLM":
        self.default_model_name = model_name
        return self

    def setup_default_emb_model_name(self, model_name: str) -> "ByzerLLM":
        self.default_emb_model_name = model_name
        return self

    def setup_default_re_rank_model_name(self, model_name: str) -> "ByzerLLM":
        self.default_rerank_model_name = model_name
        return self

    def setup(self, name: str, value: Any) -> "ByzerLLM":
        self.sys_conf[name] = value
        # update the context conf
        self.context.conf = self.sys_conf
        return self

    def setup_function_calling_format_func(self, model: str, func) -> "ByzerLLM":
        self.mapping_function_calling_format_func[model] = func
        return self

    def setup_response_class_format_func(self, model: str, func) -> "ByzerLLM":
        self.mapping_response_class_format_func[model] = func
        return self

    def setup_impl_func_format_func(self, model: str, func) -> "ByzerLLM":
        self.mapping_impl_func_format_func[model] = func
        return self

    def setup_response_class_format_after_chat_func(
        self, model: str, func
    ) -> "ByzerLLM":
        self.mapping_response_class_format_after_chat_func[model] = func
        return self

    def setup_base_system_messages(
        self, model: str, base_system_message: str
    ) -> "ByzerLLM":
        self.mapping_base_system_message[model] = base_system_message
        return self

    def setup_sys_response_class_format_func(self, model: str, func) -> "ByzerLLM":
        self.mapping_sys_response_class_format_func[model] = func
        return self

    def setup_sys_function_calling_format_func(self, model: str, func) -> "ByzerLLM":
        self.mapping_sys_function_calling_format_func[model] = func
        return self

    def setup_sys_response_class_format_after_chat_func(
        self, model: str, func
    ) -> "ByzerLLM":
        self.mapping_sys_response_class_format_after_chat_func[model] = func
        return self

    def setup_sys_impl_func_format_func(self, model: str, func) -> "ByzerLLM":
        self.mapping_sys_impl_func_format_func[model] = func
        return self

    def setup_infer_backend(self, backend: str) -> "ByzerLLM":
        self.sys_conf["infer_backend"] = backend

        if backend == InferBackend.VLLM or backend == InferBackend.DeepSpeed:
            self.sys_conf["masterMaxConcurrency"] = 1000
            self.sys_conf["workerMaxConcurrency"] = 100

        if backend == InferBackend.Transformers:
            self.sys_conf["masterMaxConcurrency"] = 1000
            self.sys_conf["workerMaxConcurrency"] = 1

        return self

    def setup_gpus_per_worker(self, num_gpus: int) -> "ByzerLLM":
        self.sys_conf["num_gpus"] = num_gpus
        return self

    def setup_cpus_per_worker(self, num_cpus: int) -> "ByzerLLM":
        self.sys_conf["num_cpus"] = num_cpus
        return self

    def setup_worker_concurrency(self, num: int) -> "ByzerLLM":
        self.sys_conf["workerMaxConcurrency"] = num
        return self

    def setup_num_workers(self, num_workers: int) -> "ByzerLLM":
        self.sys_conf["maxConcurrency"] = num_workers
        return self

    def setup_max_model_length(self, model: str, max_model_length: int) -> "ByzerLLM":
        self.mapping_max_model_length[model] = max_model_length
        return self

    def setup_max_input_length(self, model: str, max_input_length: int) -> "ByzerLLM":
        self.mapping_max_input_length[model] = max_input_length
        return self

    def setup_max_output_length(self, model: str, max_output_length: int) -> "ByzerLLM":
        self.mapping_max_output_length[model] = max_output_length
        return self

    def setup_role_mapping(
        self, model: str, role_mapping: Dict[str, str]
    ) -> "ByzerLLM":
        self.mapping_role_mapping[model] = role_mapping
        return self

    def setup_extra_generation_params(
        self, model: str, extra_generation_params: Dict[str, Any]
    ) -> "ByzerLLM":
        v = self.mapping_extra_generation_params.get(model, {})
        self.mapping_extra_generation_params[model] = {**v, **extra_generation_params}
        return self

    def setup_template(self, model: str, template: Union[Template, str]) -> "ByzerLLM":
        if template == "auto":
            meta = self.get_meta(model=model)

            is_saas_model = meta.get("model_deploy_type", None) == "saas"

            if is_saas_model:
                return self

            is_message_format = meta.get("message_format", False)

            if is_message_format:
                return self

            if "QWenLMHeadModel" in meta.get("architectures", []):
                self.setup_template(model, Templates.qwen())
                return self

            if not meta.get("support_chat_template", False):
                raise Exception(
                    f"The model({model}) is not support auto(apply chat template) for now."
                )

            self.mapping_auto_use_apply_chat_template[model] = True
            return self

        self.mapping_role_mapping[model] = template.role_mapping

        v = self.mapping_extra_generation_params.get(model, {})
        self.mapping_extra_generation_params[model] = {
            **v,
            **template.generation_config,
        }

        self.mapping_clean_func[model] = template.clean_func
        self.mapping_function_calling_format_func[model] = (
            template.function_calling_format_func
        )
        self.mapping_response_class_format_after_chat_func[model] = (
            template.response_class_format_after_chat_func
        )
        self.mapping_response_class_format_func[model] = (
            template.response_class_format_func
        )
        return self

    def sft(
        self,
        sft_name: str,
        local_data_dir_path: str,
        local_model_path: str,
        local_stage_path: str,
        pretrained_model_type: str,
        num_cpus: int,
        num_gpus: int,
        detached: bool = True,
        json_config: str = "{}",
        model_params: Dict[str, Any] = {},
        **kwargs,
    ):
        """
        finetune a pretrained model

        Args:
            sft_name (str): the uniq name of this finetune task
            local_data_dir_path (str): the local data dir path, which should contains `data.jsonl` file
            local_model_path (str): the local model path, which should contains `config.json` file
            local_stage_path (str): the local stage path which store the temp data and model
            pretrained_model_type (str): the pretrained model type, e.g. "sft/llama2","sft/baichuan"
            num_cpus (int): the number of cpus
            num_gpus (int): the number of gpus
            detached (bool, optional): whether to run this task in detached mode. Defaults to True.
            json_config (str, optional): the json config string. Defaults to "{}".
            model_params (Dict[str,Any], optional): the model params. Defaults to {}. The key should like this style `sft.int.logging_steps`, `sft.int.max_seq_length`
                                                    which contains the `sft` prefix and the type of the value.
        """
        train_params = {}
        train_params["name"] = sft_name
        train_params["data_dir"] = local_data_dir_path
        train_params["localModelDir"] = local_model_path
        train_params["pretrainedModelType"] = pretrained_model_type
        train_params["config"] = json_config
        train_params["detached"] = "true" if detached else "false"
        train_params["localPathPrefix"] = local_stage_path

        for k, v in model_params.items():
            train_params[k] = v

        sys_conf = {}
        sys_conf["num_gpus"] = num_gpus
        sys_conf["num_cpus"] = num_cpus

        r = self.raw_sft(train_params=train_params, sys_conf=sys_conf)
        if detached:
            return [i for i in r]
        return r

    def merge_lora(
        self,
        name: str,
        local_model_path: str,
        local_adpator_model_path: str,
        local_target_path: str,
    ):
        train_params = {}
        train_params["name"] = name
        train_params["modelNameOrPath"] = local_model_path
        train_params["adapterNameOrPath"] = local_adpator_model_path
        train_params["savePath"] = local_target_path
        self.raw_merge_lora(train_params=train_params, sys_conf={})
        return local_target_path

    def pretrain(
        self,
        name: str,
        local_data_dir_path: str,
        local_model_path: str,
        local_stage_path: str,
        pretrained_model_type: str,
        num_cpus: int,
        num_gpus: int,
        detached: bool = True,
        json_config: str = "{}",
        model_params: Dict[str, Any] = {},
        **kwargs,
    ):
        train_params = {}
        train_params["name"] = name
        train_params["localDataDir"] = local_data_dir_path
        train_params["localModelDir"] = local_model_path
        train_params["pretrainedModelType"] = pretrained_model_type
        train_params["deepspeedConfig"] = json_config
        train_params["detached"] = "true" if detached else "false"
        train_params["localPathPrefix"] = local_stage_path

        for k, v in model_params.items():
            train_params[k] = v

        sys_conf = {}
        sys_conf["num_gpus"] = num_gpus
        sys_conf["num_cpus"] = num_cpus

        r = self.raw_pretrain(train_params=train_params, sys_conf=sys_conf)
        if detached:
            return [i for i in r]
        return r

    def raw_sft(self, train_params: Dict[str, Any], sys_conf: Dict[str, Any] = {}):
        model_type = train_params["pretrainedModelType"].split("/")[-1]
        train_module = importlib.import_module(f"byzerllm.{model_type}")
        return train_module.sft_train([], train_params, sys_conf)

    def raw_pretrain(self, train_params: Dict[str, Any], sys_conf: Dict[str, Any] = {}):
        model_type = train_params["pretrainedModelType"][-1]
        train_module = importlib.import_module(f"byzerllm.{model_type}")
        return train_module.sfft_train([], train_params, sys_conf)

    def raw_merge_lora(self, train_params: Dict[str, Any], sys_conf: Dict[str, Any]):
        from byzerllm.utils.sft.merge_lora import merge_lora_to_base_model

        merge_lora_to_base_model([], train_params, sys_conf)

    def raw_deepspeed_to_huggingface(self, train_params: Dict[str, Any]):
        from byzerllm.utils.fulltune.pretrain.convert_to_transformers import convert

        convert(train_params, self.conf())

    def undeploy(self, udf_name: str, force: bool = False):
        import time

        try:
            model = ray.get_actor(udf_name)
            if not force:
                try:
                    meta = self.get_meta(model=udf_name)
                    if meta.get("backend", "") == "ray/vllm":
                        if "engine_placement_group_id" in meta:
                            cancel_placement_group(meta["engine_placement_group_id"])
                except Exception as inst:
                    pass
            ray.kill(model)
            if udf_name in self.meta_cache:
                del self.meta_cache[udf_name]
        except ValueError:
            pass
        time.sleep(3)

    def generate_instruction_from_history(
        self,
        model: str,
        conversations: List[Dict[str, str]],
        role_mapping: Dict[str, str] = {
            "user_role": "User:",
            "assistant_role": "Assistant:",
        },
    ):
        meta = self.get_meta(model=model)
        if self.mapping_auto_use_apply_chat_template.get(model, False) and meta.get(
            "support_chat_template", False
        ):
            return self.apply_chat_template(
                model, json.dumps(conversations, ensure_ascii=False)
            )

        new_his = []
        for item in conversations:
            if item["role"] == "system":
                value = item["content"]
                if "system_msg_func" in role_mapping:
                    value = role_mapping["system_msg_func"](
                        t=role_mapping["system_msg"], v=item["content"]
                    )
                new_his.append(value)
                continue

            if item["role"] == "user":
                value = f"{role_mapping['user_role']}{item['content']}"
                if "user_role_func" in role_mapping:
                    value = role_mapping["user_role_func"](
                        t=role_mapping["user_role"], v=item["content"]
                    )
                new_his.append(value)

            if item["role"] == "assistant":
                value = f"{role_mapping['assistant_role']}{item['content']}"
                if "user_role_func" in role_mapping:
                    value = role_mapping["assistant_role_func"](
                        t=role_mapping["assistant_role"], v=item["content"]
                    )
                new_his.append(value)

        if conversations[-1]["role"] == "user":
            new_his.append(f"{role_mapping['assistant_role']}")

        fin_ins = "\n".join(new_his)
        return fin_ins

    def is_model_exist(self, udf_name: str) -> bool:
        try:
            ray.get_actor(udf_name)
            return True
        except Exception as inst:
            return False

    def deploy(
        self,
        model_path: str,
        pretrained_model_type: str,
        udf_name: str,
        infer_params: Dict[str, Any],
    ):
        from byzerllm import common_init_model

        self.setup("UDF_CLIENT", udf_name)

        infer_backend = self.sys_conf["infer_backend"]

        if (
            infer_backend == InferBackend.VLLM
            or infer_backend == InferBackend.DeepSpeed
        ):
            if pretrained_model_type != "custom/auto":
                raise ValueError(
                    f"Backend({infer_backend}) is set. the pretrained_model_type should be `custom/auto`"
                )

        model_type = pretrained_model_type

        if pretrained_model_type.startswith("saas/"):
            model_type = pretrained_model_type.split("/")[-1]

            infer_module = importlib.import_module(f"byzerllm.saas.{model_type}")
            from byzerllm.utils.text_generator import simple_predict_func

            def init_model(
                model_refs: List[ClientObjectRef], conf: Dict[str, str]
            ) -> Any:
                from byzerllm import consume_model

                consume_model(conf)
                infer = infer_module.CustomSaasAPI(infer_params)
                return (infer, None)

            UDFBuilder.build(self.ray_context, init_model, simple_predict_func)
            return self.get_meta(model=udf_name)

        if pretrained_model_type == "bark":
            from byzerllm.bark.bark_voice import (
                build_void_infer,
                ZH_SPEAKER,
                EN_SPEAKER,
            )

            def init_model(
                model_refs: List[ClientObjectRef], conf: Dict[str, str]
            ) -> Any:
                infer = build_void_infer(
                    model_dir=model_path,
                    tokenizer_dir=f"{model_path}/pretrained_tokenizer",
                )
                return infer

            def predict_func(model, v):
                data = [json.loads(item) for item in v]
                results = [
                    {
                        "predict": model.text_to_voice(item["instruction"]).tolist(),
                        "labels": "",
                    }
                    for item in data
                ]
                return {"value": [json.dumps(results, ensure_ascii=False, indent=4)]}

            UDFBuilder.build(self.ray_context, init_model, predict_func)
            return self.get_meta(model=udf_name)

        # we put in this place so it only take effect for private model
        self.mapping_max_output_length[udf_name] = 4000

        if pretrained_model_type.startswith("custom/"):
            model_type = pretrained_model_type.split("/")[-1]

        predict_func = "simple_predict_func"
        if model_type == "chatglm2":
            predict_func = "chatglm_predict_func"

        infer_module = importlib.import_module(f"byzerllm.{model_type}")
        predict_module = importlib.import_module(f"byzerllm.utils.text_generator")

        def init_model(model_refs: List[ClientObjectRef], conf: Dict[str, str]) -> Any:
            common_init_model(model_refs, conf, model_path, is_load_from_local=True)
            model = infer_module.init_model(model_path, infer_params, conf)
            return model

        UDFBuilder.build(
            self.ray_context, init_model, getattr(predict_module, predict_func)
        )
        return self.get_meta(model=udf_name)

    def get_meta(self, model: str, llm_config: Dict[str, Any] = {}):
        if not model and not self.default_model_name:
            raise Exception("model name is required")

        if not model:
            model = self.default_model_name

        if model in self.meta_cache:
            return self.meta_cache[model]

        default_config = self.mapping_extra_generation_params.get(model, {})

        v = [{"instruction": "", "meta": True, **{**default_config, **llm_config}}]
        res = self._query(model, v)

        t = [
            LLMResponse(
                output=item["predict"],
                metadata=item.get("metadata", {}),
                input=item["input"],
            )
            for item in res
        ]

        res = {}
        if len(t) != 0 and len(t[0].output) != 0:
            res = t[0].output[0]

        self.meta_cache[model] = res
        return self.meta_cache[model]

    def tokenize(
        self, model: str, s: str, llm_config: Dict[str, Any] = {}
    ) -> List[str]:

        if not model and not self.default_model_name:
            raise Exception("model name is required")

        if not model:
            model = self.default_model_name

        default_config = self.mapping_extra_generation_params.get(model, {})

        v = [{"instruction": s, "tokenizer": True, **{**default_config, **llm_config}}]
        res = self._query(model, v)
        return [
            LLMResponse(
                output=item["predict"],
                metadata=item.get("metadata", {}),
                input=item["input"],
            )
            for item in res
        ]

    def apply_chat_template(self, model: str, s: str, llm_config: Dict[str, Any] = {}):
        if not model and not self.default_model_name:
            raise Exception("model name is required")

        if not model:
            model = self.default_model_name

        default_config = self.mapping_extra_generation_params.get(model, {})
        v = [
            {
                "instruction": s,
                "apply_chat_template": True,
                **{**default_config, **llm_config},
            }
        ]
        res = self._query(model, v)

        t = [
            LLMResponse(
                output=item["predict"],
                metadata=item.get("metadata", {}),
                input=item["input"],
            )
            for item in res
        ]
        return t[0].output

    def emb_query(self, v: str, model: str = None):
        return self.emb(model=model, request=LLMRequest(instruction=v))

    def emb(self, model, request: LLMRequest, extract_params: Dict[str, Any] = {}):

        if not model and not self.default_emb_model_name:
            raise Exception("model name is required")

        if not model:
            model = self.default_emb_model_name

        default_config = self.mapping_extra_generation_params.get(model, {})

        if isinstance(request, list):
            request = LLMRequest(instruction=request)

        if isinstance(request.instruction, str):
            v = [
                {
                    "instruction": request.instruction,
                    "embedding": True,
                    "max_length": request.max_length,
                    "top_p": request.top_p,
                    "temperature": request.temperature,
                    **default_config,
                    **extract_params,
                }
            ]
        else:
            v = [
                {
                    "instruction": x,
                    "embedding": True,
                    "max_length": request.max_length,
                    "top_p": request.top_p,
                    "temperature": request.temperature,
                    **default_config,
                    **extract_params,
                }
                for x in request.instruction
            ]
        res = self._query(model, v)

        return [
            LLMResponse(
                output=item["predict"],
                metadata=item.get("metadata", {}),
                input=item["input"],
            )
            for item in res
        ]

    def emb_rerank(
        self,
        model: str = None,
        sentence_pairs: Union[List[Tuple[str, str]], Tuple[str, str]] = [],
        extract_params: Dict[str, Any] = {},
    ) -> Union[Tuple[Tuple[str, str], float], List[Tuple[Tuple[str, str], float]]]:

        if not model and not self.default_rerank_model_name:
            raise Exception("rerank model name is required")

        if not sentence_pairs or len(sentence_pairs) == 0:
            raise Exception("rerank rerank param sentence_pairs is required")

        if not model:
            model = self.default_rerank_model_name

        default_config = self.mapping_extra_generation_params.get(model, {})

        v = [
            {
                "instruction": sentence_pairs,
                "embedding": True,
                "embed_rerank": True,
                **default_config,
                **extract_params,
            }
        ]
        res = self._query(model, v)

        return [
            LLMResponse(
                output=item["predict"],
                metadata=item.get("metadata", {}),
                input=item["input"],
            )
            for item in res
        ]

    def _generate_ins(
        self, model: str, request: LLMRequest, role_mapping: Dict[str, str]
    ):
        if not role_mapping["user_role"]:
            return request.instruction

        sys_msg = role_mapping["system_msg"]
        if "system_msg_func" in role_mapping:
            sys_msg = "You are a helpful assistant. Think it over and answer the user question correctly."

        conversations = [{"role": "system", "content": sys_msg}]
        # conversations += [{"role":item.role,"content":item.content} for item in request.extra_params.history]

        conversations += self._to_openai_format(request=request)

        final_ins = self.generate_instruction_from_history(
            model, conversations, role_mapping
        )

        return final_ins

    def _to_openai_format(self, request: LLMRequest):
        conversations = []
        if isinstance(request.instruction, str):
            conversations += [{"role": "user", "content": request.instruction}]
        else:
            conversations += [
                {"role": "user", "content": x} for x in request.instruction
            ]
        return conversations

    def execute_function_calling(
        self, response: LLMResponse, tools: List[Callable], func_params: Dict[str, Any]
    ) -> LLMFunctionCallResponse:

        r = LLMFunctionCallResponse(
            response=response, values=[], metadata={"reason": ""}
        )

        is_json = False
        try:
            json.loads(response.output)
            is_json = True
        except Exception as inst:
            pass

        code = response.output

        if not is_json:
            if code.strip().startswith("```json"):
                index = code.rfind("```")
                if index != -1:
                    code = code.strip()[7 : index - 1]
                    try:
                        json.loads(code)
                        is_json = True
                    except Exception as inst:
                        pass

        if not is_json:
            codes = code_utils.extract_code(response.output)
            if len(codes) == 0:
                r.metadata["reason"] = "No json block found"
                return r

            lang, code = codes[-1]

            if lang != "json":
                r.metadata["reason"] = "No json block found"
                return r

        try:
            temp = json.loads(code)
            if isinstance(temp, list):
                temp = temp[-1]
            ms = FunctionCallList.parse_obj(temp)
        except Exception as inst:
            r.metadata["reason"] = str(inst) + "\n" + traceback.format_exc()
            return r

        _func_maps = dict([(t.__name__, t) for t in tools])

        if func_params is None:
            func_params = {}

        try:
            r.metadata["selected_functions"] = []
            for m in ms.tool_calls:
                if m.function.name in _func_maps:
                    r.metadata["selected_functions"].append(m.function.name)
                    r.values.append(
                        _func_maps[m.function.name](
                            **m.function.arguments, **func_params
                        )
                    )
        except Exception as inst:
            r.metadata["reason"] = str(inst) + "\n" + traceback.format_exc()

        return r

    def execute_generate_func(
        self,
        func_name: str,
        impl_func_params: Optional[Dict[str, Any]],
        response: LLMResponse,
        response_class: pydantic.BaseModel,
    ) -> LLMClassResponse:

        r = LLMClassResponse(response=response, value=None, metadata={"reason": ""})

        is_python_code = False
        if code_utils.infer_lang(response.output) == "python":
            is_python_code = True

        code = response.output

        if not is_python_code:
            codes = code_utils.extract_code(response.output)

            if len(codes) == 0:
                r.metadata["reason"] = "No Python block found"
                return r

            lang, code = codes[-1]

            if lang != "python":
                r.metadata["reason"] = "No Python block found"
                return r

        (status, output, variables) = exec_capture_output(code, {func_name: True})
        if status != 0:
            r.metadata["reason"] = output
            return r

        try:
            if impl_func_params is None:
                impl_func_params = {}
            res_json = variables[func_name](**impl_func_params)
            r.metadata["raw_func"] = code
            r.metadata["func"] = variables[func_name]
            if isinstance(res_json, str):
                res_json = json.loads(res_json)
            r.value = response_class.parse_obj(res_json)
        except Exception as inst:
            r.metadata["reason"] = str(inst) + "\n" + traceback.format_exc()
            return r

        return r

    def execute_response_format(
        self, response: LLMResponse, response_class: pydantic.BaseModel
    ):

        r = LLMClassResponse(response=response, value=None, metadata={"reason": ""})
        is_json = False
        try:
            json.loads(response.output)
            is_json = True
        except Exception as inst:
            pass

        code = response.output

        if not is_json:
            if code.strip().startswith("```json"):
                index = code.rfind("```")
                if index != -1:
                    code = code.strip()[7 : index - 1]
                    try:
                        json.loads(code)
                        is_json = True
                    except Exception as inst:
                        pass

        if not is_json:
            codes = code_utils.extract_code(response.output)
            if len(codes) == 0:
                r.metadata["reason"] = "No json block found"
                return r

            lang, code = codes[-1]

            if lang != "json":
                r.metadata["reason"] = "No json block found"
                return r

        try:
            try:
                obj = json.loads(code)
            except Exception as inst:
                print(
                    "Fail to parse json. Error:\n"
                    + str(inst)
                    + "\n"
                    + traceback.format_exc(),
                    flush=True,
                )
                obj = json.loads(repair_json_str(code))
            ms = response_class.parse_obj(obj)
        except Exception as inst:
            r.metadata["reason"] = str(inst) + "\n" + traceback.format_exc()
            return r

        r.value = ms

        return r

    def require_template(self, model: str):
        meta = self.get_meta(model=model)
        is_saas_model = meta.get("model_deploy_type", None) == "saas"
        is_message_format = meta.get("message_format", False)
        support_chat_template = meta.get("support_chat_template", False)
        return not is_saas_model and not is_message_format and not support_chat_template

    def abort(self, request_id: str, model: Optional[str] = None):
        if not model and not self.default_model_name:
            raise Exception("model name is required")
        if not model:
            model = self.default_model_name

        meta = self.get_meta(model=model)
        if meta.get("backend", None) != "ray/vllm":
            raise Exception("abort only support ray/vllm backend")

        self.chat_oai(
            conversations=[{"role": "user", "content": f"{request_id}"}],
            llm_config={"gen.request_id": request_id, "gen.abort": True},
        )

    def chat_oai(
        self,
        conversations,
        tools: List[Union[Callable, str]] = [],
        tool_choice: Optional[Union[Callable, str]] = None,
        execute_tool: bool = False,
        impl_func: Optional[Callable] = None,
        execute_impl_func: bool = False,
        impl_func_params: Optional[Dict[str, Any]] = None,
        func_params: Optional[Dict[str, Any]] = None,
        response_class: Optional[Union[pydantic.BaseModel, str]] = None,
        response_after_chat: Optional[Union[pydantic.BaseModel, str]] = False,
        enable_default_sys_message: bool = True,
        model: Optional[str] = None,
        role_mapping=None,
        llm_config: Dict[str, Any] = {},
        only_return_prompt: bool = False,
        extra_request_params:Dict[str,Any] = {}
    ) -> Union[
        List[LLMResponse], List[LLMFunctionCallResponse], List[LLMClassResponse]
    ]:

        if not self.default_model_name and not model:
            raise Exception(
                "Use llm.setup_default_model_name to setup default model name or setup the model parameter"
            )

        if not model:
            model = self.default_model_name

        if role_mapping is None:
            role_mapping = self.mapping_role_mapping.get(
                model, self.default_role_mapping
            )

        if response_class and (tools or tool_choice):
            raise Exception(
                "function calling is enabled,response_class should not be set."
            )

        if impl_func and not response_class:
            raise Exception("impl_func is enabled,response_class should be set.")

        if isinstance(conversations, str):
            conversations = [{"role": "user", "content": conversations}]

        if enable_default_sys_message:
            first_message = conversations[0]
            base_abilities = []
            if response_class:
                base_abilities.append(BaseAbility.RESPONSE_WITH_CLASS)
            if impl_func:
                base_abilities.append(BaseAbility.FUNCTION_IMPL)
            if tools or tool_choice:
                base_abilities.append(BaseAbility.FUNCTION_CALLING)

            if base_abilities and first_message["role"] == "user":
                conversations.insert(
                    0,
                    {
                        "role": "system",
                        "content": self.mapping_base_system_message.get(
                            model, base_ability_format(base_abilities=base_abilities)
                        ),
                    },
                )

            if first_message["role"] == "system":
                first_message[
                    "content"
                ] = f"""{self.mapping_base_system_message.get(model,base_ability_format(base_abilities=base_abilities))}
{first_message["content"]}"""

        meta = self.get_meta(model=model)
        is_saas_model = meta.get("model_deploy_type", None) == "saas"
        is_message_format = meta.get("message_format", False)

        temp_conversations = copy.deepcopy(conversations)
        last_message = temp_conversations[-1]

        # function calling
        if tools or tool_choice:
            f = (
                self.mapping_function_calling_format_func.get(
                    model, function_calling_format
                )
                if not enable_default_sys_message
                else self.mapping_sys_function_calling_format_func.get(
                    model, sys_function_calling_format
                )
            )
            last_message["content"] = f(last_message["content"], tools, tool_choice)

        # implement function and the function should return a response class
        elif impl_func and response_class:
            f = (
                self.mapping_impl_func_format_func.get(model, function_impl_format)
                if not enable_default_sys_message
                else self.mapping_sys_impl_func_format_func.get(
                    model, sys_function_impl_format
                )
            )
            last_message["content"] = f(
                last_message["content"], impl_func, cls=response_class
            )

        # generate response class
        elif response_class and not response_after_chat:
            f = (
                self.mapping_response_class_format_func.get(
                    model, response_class_format
                )
                if not enable_default_sys_message
                else self.mapping_sys_response_class_format_func.get(
                    model, sys_response_class_format
                )
            )
            last_message["content"] = f(last_message["content"], cls=response_class)

        if is_saas_model or is_message_format:
            final_ins = last_message["content"]
            history = []
            for item in temp_conversations[:-1]:
                # clean metadata field in conversation
                # which may used by agent.
                if "metadata" in item:
                    del item["metadata"]
                history.append(item)

        else:
            final_ins = self.generate_instruction_from_history(
                model, temp_conversations, role_mapping
            )
            history = []

        default_config = self.mapping_extra_generation_params.get(model, {})

        if self.get_max_output_length(model) > 0:
            default_config["max_length"] = self.get_max_output_length(model)

        v = [
            {
                "instruction": final_ins,
                "history": history,
                "extra_request_params": extra_request_params,
                **default_config,
                **llm_config,                    
            }
        ]

        if only_return_prompt:
            responses = [
                LLMResponse(output="", metadata=item, input=item["instruction"])
                for item in v
            ]
            if response_class or response_after_chat:
                new_responses = []
                for response in responses:
                    temp = LLMClassResponse(
                        response=response,
                        value=response,
                        metadata={"reason": "Only return prompt"},
                    )
                    new_responses.append(temp)
                return new_responses
            return responses

        res = self._query(model, v)
        clean_func = self.mapping_clean_func.get(model, lambda s: s)

        responses = [
            LLMResponse(
                output=clean_func(item["predict"]),
                metadata=item.get("metadata", {}),
                input=item["input"],
            )
            for item in res
        ]

        ## handle impl_func response
        if impl_func and response_class and execute_impl_func:
            final_result = []
            for response in responses:
                final_result.append(
                    self.execute_generate_func(
                        func_name=impl_func.__name__,
                        impl_func_params=impl_func_params or func_params,
                        response=response,
                        response_class=response_class,
                    )
                )
            return final_result

        if impl_func and response_class:
            return responses

        ## handle response_class response
        temp_result = responses
        if response_class and response_after_chat:
            temp_result = []
            f = self.mapping_response_class_format_after_chat_func.get(
                model, response_class_format_after_chat
            )
            for response in responses:
                new_conversations = temp_conversations + [
                    {"content": response.output, "role": "assistant"},
                    {"content": f(response_class), "role": "user"},
                ]
                temp_result.append(
                    self.chat_oai(
                        new_conversations,
                        role_mapping=role_mapping,
                        llm_config=llm_config,
                    )[0]
                )

        if response_class:
            final_result = []
            for response in temp_result:
                final_result.append(
                    self.execute_response_format(
                        response=response, response_class=response_class
                    )
                )
            return final_result

        ## handle function calling response
        if execute_tool:
            final_result = []
            for response in responses:
                final_result.append(
                    self.execute_function_calling(
                        response=response, tools=tools, func_params=func_params
                    )
                )

            return final_result

        return responses

    def stream_chat_oai(
        self,
        conversations,
        model: Optional[str] = None,
        role_mapping=None,
        delta_mode: bool = False,
        llm_config: Dict[str, Any] = {},
        extra_request_params:Dict[str,Any] = {}
    ):

        if not model:
            model = self.default_model_name

        meta = self.get_meta(model=model)
        if not meta.get("support_stream", False):
            raise Exception(f"The model({model}) is not support stream chat for now.")

        v = self.chat_oai(
            conversations,
            model=model,
            role_mapping=role_mapping,
            llm_config={**llm_config, **{"generation.stream": True}},
            extra_request_params=extra_request_params
        )
        request_id = v[0].metadata["request_id"]
        stream_server_type = v[0].metadata.get("stream_server", "VLLM_STREAM_SERVER")
        server = ray.get_actor(stream_server_type)

        pre_reasoning_text = ""
        pre_generated_text = ""

        while True:
            final_output = ray.get(server.get_item.remote(request_id))
            if isinstance(final_output, str):
                time.sleep(0.01)
                continue

            if final_output is None:
                break

            if stream_server_type == "BlockBinaryStreamServer":
                binary_data = final_output.outputs[0].text
                yield (binary_data, final_output.outputs[0].metadata)
            else:
                text_outputs = final_output.outputs
                clean_func = self.mapping_clean_func.get(model, lambda s: s)
                generated_text = text_outputs[0].text
                metadata = text_outputs[0].metadata
                reasoning_text = metadata.reasoning_content or ""
                if (
                    not pre_generated_text 
                    and generated_text == pre_generated_text
                    and not pre_reasoning_text 
                    and reasoning_text == pre_reasoning_text
                ):
                    continue

                if delta_mode and (pre_generated_text or pre_reasoning_text):
                    s = generated_text[len(pre_generated_text) :]
                    metadata.reasoning_content = reasoning_text[len(pre_reasoning_text) :]
                else:
                    s = generated_text                    
                pre_generated_text = generated_text
                pre_reasoning_text = reasoning_text
                yield (clean_func(s), metadata)

    async def async_stream_chat_oai(
        self,
        conversations,
        role_mapping=None,
        model: Optional[str] = None,
        delta_mode: bool = False,
        llm_config: Dict[str, Any] = {},
        extra_request_params:Dict[str,Any] = {}
    ):

        if not model:
            model = self.default_model_name

        meta = self.get_meta(model=model)
        if not meta.get("support_stream", False):
            raise Exception(f"The model({model}) is not support stream chat for now.")

        v = self.chat_oai(
            conversations,
            model=model,
            role_mapping=role_mapping,
            llm_config={**llm_config, **{"generation.stream": True}},
            extra_request_params=extra_request_params
        )
        request_id = v[0].metadata["request_id"]
        stream_server_type = v[0].metadata.get("stream_server", "VLLM_STREAM_SERVER")
        server = ray.get_actor(stream_server_type)

        pre_generated_text = ""
        pre_reasoning_text = ""
        while True:
            final_output = await server.get_item.remote(request_id)
            if isinstance(final_output, str):
                await asyncio.sleep(0.01)
                continue

            if final_output is None:
                break

            if stream_server_type == "BlockBinaryStreamServer":
                binary_data = final_output.outputs[0].text
                yield (binary_data, final_output.outputs[0].metadata)
            else:
                text_outputs = final_output.outputs
                clean_func = self.mapping_clean_func.get(model, lambda s: s)
                generated_text = text_outputs[0].text
                metadata = text_outputs[0].metadata
                reasoning_text = metadata.reasoning_content or ""
                if (
                    not pre_generated_text 
                    and generated_text == pre_generated_text
                    and not pre_reasoning_text 
                    and reasoning_text == pre_reasoning_text
                ):
                    continue

                if delta_mode and (pre_generated_text or pre_reasoning_text):
                    s = generated_text[len(pre_generated_text) :]
                    metadata.reasoning_content = reasoning_text[len(pre_reasoning_text) :]
                else:
                    s = generated_text
                pre_generated_text = generated_text
                pre_reasoning_text = reasoning_text
                yield (clean_func(s), metadata)

    def clear_impl_cache(
        self,
        model: Optional[str] = None,
        full_func_name: Optional[str] = None,
        instruction: Optional[str] = None,
    ):
        if model is None and full_func_name is None and instruction is None:
            self.func_impl_cache = {}

        if model is not None and full_func_name is not None and instruction is None:
            raise Exception("instruction is required")

        if model is not None:
            instruction = "" if not instruction else instruction
            full_func_name = "" if not full_func_name else full_func_name

            key = f"{model}_{instruction}_{full_func_name}"
            for k in list(self.func_impl_cache.keys()):
                if k.startswith(key):
                    del self.func_impl_cache[k]
            return self

        if full_func_name is not None:
            instruction = "" if not instruction else instruction
            model = "" if not model else model
            key = f"{model}_{instruction}_{full_func_name}"
            for k in list(self.func_impl_cache.keys()):
                if k.endswith(key):
                    del self.func_impl_cache[k]
            return self

    def prompt(
        self,
        model: Optional[str] = None,
        render: Optional[str] = "jinja2",
        check_result: bool = False,
        options: Dict[str, Any] = {},
        return_origin_response: bool = False,
        marker: Optional[str] = None,
        assistant_prefix: Optional[str] = None,
        meta_holder: Optional[Any] = None,
        conversation: List[Dict[str,Any]] = []
    ):
        if model is None:
            if "model" in options:
                model = options.pop("model")
            else:
                model = self.default_model_name

        def is_instance_of_generator(v):
            from typing import Generator, get_origin, get_args
            import collections

            if get_origin(v) is collections.abc.Generator:
                args = get_args(v)
                if args == (str, type(None), type(None)):
                    return True
            return False

        def _impl(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                signature = inspect.signature(func)
                arguments = signature.bind(*args, **kwargs)
                arguments.apply_defaults()
                input_dict = {}
                for param in signature.parameters:
                    input_dict.update({param: arguments.arguments[param]})

                if "self" in input_dict:
                    instance = input_dict.pop("self")
                    new_input_dic = func(instance, **input_dict)
                    if new_input_dic and not isinstance(new_input_dic, dict):
                        raise TypeError(
                            f"Return value of {func.__name__} should be a dict"
                        )
                    if new_input_dic:
                        input_dict = {**input_dict, **new_input_dic}
                else:
                    new_input_dic = func(**input_dict)
                    if new_input_dic and not isinstance(new_input_dic, dict):
                        raise TypeError(
                            f"Return value of {func.__name__} should be a dict"
                        )
                    if new_input_dic:
                        input_dict = {**input_dict, **new_input_dic}

                prompt_str = format_prompt_jinja2(func, **input_dict)

                if marker:
                    prompt_str = f"{prompt_str}\n\n{marker}"

                if is_instance_of_generator(signature.return_annotation):
                    temp_options = {**{"delta_mode": True}, **options}
                    conversations = self.conversation + [{"role": "user", "content": prompt_str}]
                    if assistant_prefix:
                        conversations = conversations + [{"role": "assistant", "content": assistant_prefix}]

                    t = self.stream_chat_oai(
                        conversations=conversations,
                        model=model,
                        **temp_options,
                    )
                    
                    if return_origin_response:
                        return t
                    
                    def generator():
                        for item,meta in t:
                            if meta_holder and meta:
                                meta_holder.meta = meta
                            yield item                    
                    return generator()

                if issubclass(signature.return_annotation, pydantic.BaseModel):
                    response_class = signature.return_annotation
                    conversations = self.conversation + [{"role": "user", "content": prompt_str}]
                    if assistant_prefix:
                        conversations = conversations + [{"role": "assistant", "content": assistant_prefix}]
                    t = self.chat_oai(
                        model=model,
                        conversations=conversations,
                        response_class=response_class,
                        impl_func_params=input_dict,
                        **options,
                    )
                    
                    
                    if meta_holder and t[0].metadata:
                        meta_holder.meta = t[0].metadata

                    if return_origin_response:
                        return t
                    r: LLMClassResponse = t[0]
                    
                    if r.value is None and check_result:
                        logger.warning(
                            f"""
                                {func.__name__} return None.
                                metadata:
                                {r.metadata}
                                response:
                                {r.response}
                            """
                        )
                    return r.value
                elif issubclass(signature.return_annotation, str):
                    conversations = self.conversation + [{"role": "user", "content": prompt_str}]
                    if assistant_prefix:
                        conversations = conversations + [{"role": "assistant", "content": assistant_prefix}]                        
                    
                    t = self.chat_oai(
                        model=model,
                        conversations=conversations,
                        **options,
                    )

                    if meta_holder and t[0].metadata:
                        meta_holder.meta = t[0].metadata 

                    if return_origin_response:
                        return t
                    return t[0].output
                else:
                    raise Exception(
                        f"{func.__name__} should return a pydantic model or string"
                    )

            return wrapper

        return _impl

    def response(
        self,
        instruction: Optional[str] = None,
        model: Optional[str] = None,
        verbose: Optional[bool] = None,
    ):
        if model is None:
            model = self.default_model_name
        if instruction is None:
            instruction = ""

        if verbose is None:
            verbose = self.verbose

        def _impl(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                signature = inspect.signature(func)
                arguments = signature.bind(*args, **kwargs)
                arguments.apply_defaults()
                input_dict = {}
                for param in signature.parameters:
                    input_dict.update({param: arguments.arguments[param]})

                if len(input_dict.keys()) != 1:
                    raise Exception(
                        "response function should have only one parameter which type should be string"
                    )

                if issubclass(signature.return_annotation, pydantic.BaseModel):
                    response_class = signature.return_annotation
                else:
                    raise Exception("impl function should return a pydantic model")

                start_time = time.monotonic()

                t = self.chat_oai(
                    model=model,
                    conversations=[
                        {"role": "user", "content": list(input_dict.values())[0]}
                    ],
                    response_class=response_class,
                    impl_func_params=input_dict,
                )

                r: LLMClassResponse = t[0]

                if verbose:
                    print(
                        f"""cost {time.monotonic() - start_time} seconds""", flush=True
                    )

                return r.value

            return wrapper

        return _impl

    def impl(
        self,
        instruction: Optional[str] = None,
        model: Optional[str] = None,
        verbose: Optional[bool] = None,
        skip_cache: bool = False,
    ):
        if model is None:
            model = self.default_model_name
        if instruction is None:
            instruction = ""

        if verbose is None:
            verbose = self.verbose

        def _impl(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):

                key = f"{model}_{instruction}_{func.__module__}.{func.__name__}"
                signature = inspect.signature(func)
                arguments = signature.bind(*args, **kwargs)
                arguments.apply_defaults()

                if issubclass(signature.return_annotation, pydantic.BaseModel):
                    response_class = signature.return_annotation
                else:
                    raise Exception("impl function should return a pydantic model")

                if not skip_cache and key in self.func_impl_cache:
                    if verbose:
                        print(f""" {key} in cache, skip impl function""")
                    return response_class.parse_obj(
                        self.func_impl_cache[key](*args, **kwargs)
                    )

                input_dict = {}
                for param in signature.parameters:
                    input_dict.update({param: arguments.arguments[param]})

                start_time = time.monotonic()

                t = self.chat_oai(
                    model=model,
                    conversations=[{"role": "user", "content": instruction}],
                    impl_func=func,
                    response_class=response_class,
                    execute_impl_func=True,
                    impl_func_params=input_dict,
                )

                r: LLMClassResponse = t[0]

                if verbose:
                    print(
                        f"""Generate code for {key}: 
```python
{r.metadata["raw_func"]}
``` 
cost {time.monotonic() - start_time} seconds                     
""",
                        flush=True,
                    )

                if not skip_cache and key not in self.func_impl_cache:
                    self.func_impl_cache[key] = r.metadata["func"]

                return r.value

            return wrapper

        return _impl

    def raw_chat(
        self,
        model,
        request: Union[LLMRequest, str],
        extract_params: Dict[str, Any] = {},
    ) -> List[LLMResponse]:
        if isinstance(request, str):
            request = LLMRequest(instruction=request)

        return self.chat(model, request, extract_params)

    def chat(
        self,
        model,
        request: Union[LLMRequest, str],
        extract_params: Dict[str, Any] = {},
    ) -> List[LLMResponse]:
        if not model and not self.default_model_name:
            raise Exception("model name is required")

        if not model:
            model = self.default_model_name

        default_config = self.mapping_extra_generation_params.get(model, {})

        default_role_mapping = self.mapping_role_mapping.get(
            model, self.default_role_mapping
        )

        if isinstance(request, str):
            request = LLMRequest(instruction=request)

        if isinstance(request.instruction, str):

            final_input = self._generate_ins(model, request, default_role_mapping)

            v = [
                {
                    "instruction": final_input,
                    "max_length": request.max_length,
                    "top_p": request.top_p,
                    "temperature": request.temperature,
                    **default_config,
                    **extract_params,
                }
            ]
        else:
            v = []
            for x in request.instruction:

                new_request = LLMRequest(
                    instruction=x,
                    embedding=request.embedding,
                    max_length=request.max_length,
                    top_p=request.top_p,
                    temperature=request.temperature,
                )

                final_input = self._generate_ins(
                    model, new_request, default_role_mapping
                )

                v.append(
                    {
                        "instruction": final_input,
                        "max_length": request.max_length,
                        "top_p": request.top_p,
                        "temperature": request.temperature,
                        **default_config,
                        **extract_params,
                    }
                )
        res = self._query(model, v)
        clean_func = self.mapping_clean_func.get(model, lambda s: s)
        return [
            LLMResponse(
                output=clean_func(item["predict"]),
                metadata=item.get("metadata", {}),
                input=item["input"],
            )
            for item in res
        ]

    def apply_sql_func(
        self,
        sql: str,
        data: List[Dict[str, Any]],
        owner: str = "admin",
        url: str = "http://127.0.0.1:9003/model/predict",
    ):
        if self.byzer_engine_url and url == "http://127.0.0.1:9003/model/predict":
            url = self.byzer_engine_url
        res = self._rest_byzer_engine(sql, data, owner, url)
        return res

    def _rest_byzer_script(
        self, sql: str, owner: str, url: str = "http://127.0.0.1:9003/run/script"
    ):
        import requests
        import json

        data = {
            "sessionPerUser": "true",
            "sessionPerRequest": "true",
            "owner": owner,
            "sql": sql,
            "includeSchema": True,
        }
        response = requests.post(url, data=data)

        if response.status_code != 200:
            raise Exception(
                f"{self.url} status:{response.status_code} content: {response.text} request: json/{json.dumps(data,ensure_ascii=False)}"
            )
        res = json.loads(response.text)
        return res

    def _rest_byzer_engine(
        self, sql: str, table: List[Dict[str, Any]], owner: str, url: str
    ):
        import requests
        import json

        data = {
            "sessionPerUser": "true",
            "sessionPerRequest": "true",
            "owner": owner,
            "dataType": "row",
            "sql": sql,
            "data": json.dumps(table, ensure_ascii=False),
        }
        response = requests.post(url, data=data)

        if response.status_code != 200:
            raise Exception(
                f"{self.url} status:{response.status_code} content: {response.text} request: json/{json.dumps(data,ensure_ascii=False)}"
            )
        res = json.loads(response.text)
        return res[0]

    def get_max_model_length(self, model: str):
        return self.mapping_max_model_length.get(model, None)

    def get_max_output_length(self, model: str):
        return self.mapping_max_output_length.get(model, self.default_max_output_length)

    def get_max_input_length(self, model: str):
        return self.mapping_max_input_length.get(model, None)

    def _query(self, model: str, input_value: List[Dict[str, Any]]):
        
        if not self.skip_nontext_check:
            try:
                from byzerllm.utils.nontext import Image, Audio

                for v in input_value:
                    s = v["instruction"]
                    image = Image(s)
                    if image.has_image():
                        c = image.to_content()
                        v["instruction"] = json.dumps(c, ensure_ascii=False)

                    audio = Audio(s)
                    if audio.has_audio():
                        c = audio.to_content()
                        v["instruction"] = json.dumps(c, ensure_ascii=False)

            except Exception as inst:
                pass

        event_result = self._trigger_event(
            EventName.BEFORE_CALL_MODEL, self, model, input_value
        )
        if event_result is not None:
            return event_result

        udf_master = ray.get_actor(model)

        try:
            new_input_value = [json.dumps(x, ensure_ascii=False) for x in input_value]
        except Exception as inst:
            raise Exception(
                f"input_value should be json serializable, got {input_value}"
            )

        if self.verbose:
            print(f"Send to model[{model}]:{new_input_value}")

        index = -1
        try:
            worker_id = -1
            if self.pin_model_worker_mapping:
                if input_value[0].get("embedding", False):
                    worker_id = self.pin_model_worker_mapping.get("embedding", -1)
                elif input_value[0].get("tokenizer", False):
                    worker_id = self.pin_model_worker_mapping.get("tokenizer", -1)
                elif input_value[0].get("apply_chat_template", False):
                    worker_id = self.pin_model_worker_mapping.get(
                        "apply_chat_template", -1
                    )
                elif input_value[0].get("meta", False):
                    worker_id = self.pin_model_worker_mapping.get("meta", -1)            

            [index, worker] = ray.get(udf_master.get.remote(worker_id))
            res = ray.get(worker.async_apply.remote(new_input_value))

            event_result = self._trigger_event(
                EventName.AFTER_CALL_MODEL, self, model, json.loads(res["value"][0])
            )
            if event_result is not None:
                return event_result

            return json.loads(res["value"][0])
        except Exception as inst:
            raise inst
        finally:
            if index != -1:
                ray.get(udf_master.give_back.remote(index))
