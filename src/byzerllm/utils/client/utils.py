from . import code_utils
import json
from typing import Any, TypeVar, Dict,Union,List
from ...utils import prompts as PROMPTS



def is_summary(data_analysis,prompt:str,role_mapping:Dict[str,str])->bool:
    from byzerllm.utils.client import LLMRequest,LLMRequestExtra
    p = PROMPTS.prompt_is_summary(prompt)
    v = data_analysis.llm.chat(None, request=LLMRequest(instruction=p,extra_params=LLMRequestExtra(**role_mapping)))[0].output
    is_summary = code_utils.get_value_from_llm_str(v,"is_summary",True)    
    return is_summary
    

def is_visualization(data_analysis,prompt:str,role_mapping:Dict[str,str])->bool:
    from byzerllm.utils.client import LLMRequest,LLMRequestExtra
    p = PROMPTS.prompt_is_visualization(prompt)    
    v = data_analysis.llm.chat(None, request=LLMRequest(instruction=p,extra_params=LLMRequestExtra(**role_mapping)))[0].output
    is_visualization = code_utils.get_value_from_llm_str(v,"is_visualization",False)
    return is_visualization 

def should_generate_code_to_response(data_analysis,prompt:str,role_mapping:Dict[str,str]):  
    from byzerllm.utils.client import LLMRequest,LLMRequestExtra  
    preview_csv = data_analysis.file_preview
    p = PROMPTS.prompt_should_generate_code_to_response(data_analysis.file_path,prompt,preview_csv)       
    v = data_analysis.llm.chat(None,request=LLMRequest(instruction=p,extra_params=LLMRequestExtra(**role_mapping)))[0].output
    need_code = code_utils.get_value_from_llm_str(v,"need_code",True)
    return need_code  
    