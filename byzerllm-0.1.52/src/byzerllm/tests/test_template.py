from byzerllm.utils.client import ByzerLLM,InferBackend,Templates
from typing import Dict,Any
import types

def get_meta(self,model:str,llm_config:Dict[str,Any]={}):        
    return {}

def test_qwen_template():
    model = "chat"
    llm = ByzerLLM(verbose=True)
    llm.setup_default_model_name(model)
    llm.setup_template(model,Templates.qwen())

    llm.get_meta = types.MethodType(get_meta, llm)     

    config = llm.mapping_extra_generation_params.get(model,{})  
        
    role_mapping =llm.mapping_role_mapping.get(model, llm.default_role_mapping)  

    v = llm.generate_instruction_from_history(model,[{
        "content":"你好",
        "role":"user"
    }],role_mapping=role_mapping)    
    print(v,flush=True)
    assert v == '''<|im_start|>user
你好
<|im_end|>
<|im_start|>assistant
'''

def test_llama_template():
    model = "chat"
    llm = ByzerLLM(verbose=True)
    llm.setup_default_model_name(model)
    llm.setup_template(model,Templates.llama())
    llm.get_meta = types.MethodType(get_meta, llm)

    config = llm.mapping_extra_generation_params.get(model,{})  
        
    role_mapping =llm.mapping_role_mapping.get(model, llm.default_role_mapping)  

    v = llm.generate_instruction_from_history(model,[{
        "content":"你好",
        "role":"user"
    }],role_mapping=role_mapping)    
    print(v,flush=True)
    assert v == '''<s>[INST] 你好 [/INST]\n'''

def test_default_template():
    model = "chat"
    llm = ByzerLLM(verbose=True)
    llm.setup_default_model_name(model)
    # llm.setup_template(model,Templates.qwen())
    llm.get_meta = types.MethodType(get_meta, llm)

    config = llm.mapping_extra_generation_params.get(model,{})  
        
    role_mapping =llm.mapping_role_mapping.get(model, llm.default_role_mapping)  

    v = llm.generate_instruction_from_history(model,[{
        "content":"你好",
        "role":"user"
    }],role_mapping=role_mapping)    
    print(v,flush=True)
    assert v == '''User:你好
Assistant:'''

def test_default_system_msg_template():
    model = "chat"
    llm = ByzerLLM(verbose=True)
    llm.setup_default_model_name(model)
    # llm.setup_template(model,Templates.qwen())
    llm.get_meta = types.MethodType(get_meta, llm)

    config = llm.mapping_extra_generation_params.get(model,{})  
        
    role_mapping =llm.mapping_role_mapping.get(model, llm.default_role_mapping)  

    v = llm.generate_instruction_from_history(model,[{
        "content":"you are a help full assistant",
        "role":"system"
    },
        {
        "content":"你好",
        "role":"user"
    }],role_mapping=role_mapping)    
    print(v,flush=True)
    assert v == '''you are a help full assistant
User:你好
Assistant:'''

def test_qwen_sys_msg_template():
    model = "chat"
    llm = ByzerLLM(verbose=True)
    llm.setup_default_model_name(model)
    llm.setup_template(model,Templates.qwen())
    llm.get_meta = types.MethodType(get_meta, llm)

    config = llm.mapping_extra_generation_params.get(model,{})  
        
    role_mapping =llm.mapping_role_mapping.get(model, llm.default_role_mapping)  

    v = llm.generate_instruction_from_history(model,[{
        "content":"you are a help full assistant",
        "role":"system"
    },
        {
        "content":"你好",
        "role":"user"
    }],role_mapping=role_mapping)    
    print(v,flush=True) 
    assert v == '''<|im_start|>system
you are a help full assistant<|im_end|>
<|im_start|>user
你好
<|im_end|>
<|im_start|>assistant
'''   
    
    

