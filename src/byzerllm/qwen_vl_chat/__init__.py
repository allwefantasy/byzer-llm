from transformers import AutoModelForCausalLM, AutoTokenizer,GenerationConfig
import json
import os
import io
from typing import Any,Any,Dict, List,Tuple,Generator
import base64
import uuid
import tempfile

def get_meta(self): 
    config = self.config   
    return [{
        "model_deploy_type": "proprietary",
        "backend":"transformers",
        "max_model_len":getattr(config, "model_max_length", -1),
        "architectures":getattr(config, "architectures", []),
        "message_format":True,
    }]

def stream_chat(self,tokenizer,ins:str, his:List[Dict[str,str]]=[],  
        max_length:int=4096, 
        top_p:float=0.95,
        temperature:float=0.1,**kwargs):
    image = kwargs["image"]
    image_b = base64.b64decode(image)

    temp_image_dir = os.path.join(tempfile.gettempdir(),"byzerllm","visualglm","images")
    if "temp_image_dir" in kwargs:
        temp_image_dir = kwargs["temp_image_dir"]

    if not os.path.exists(temp_image_dir):
        os.makedirs(temp_image_dir)

    image_file = os.path.join(temp_image_dir,f"{str(uuid.uuid4())}.jpg")
    
    if "input_image_path" in kwargs:
        image_file = kwargs["input_image_path"]

    with open(image_file,"wb") as f:
        f.write(image_b)

    # history format [('Picture 1:<img>https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg</img>\n这是什么?',
    # '图中是一名女子和她的狗在沙滩上玩耍，狗的品种应该是一只拉布拉多犬。')]     
    input_history = None
    if "history" in kwargs:
        input_history = []
        temp_input_history = json.loads(kwargs["history"]) 
        temp_input_history_length = len(temp_input_history)
        for i in range(0,temp_input_history_length,2):
            input_history.append((temp_input_history[i]["content"],temp_input_history[i+1]["content"]))
            
    if not input_history:           
        query = tokenizer.from_list_format([
        {'image': image_file}, 
        {'text': ins},])
        response, history = self.chat(tokenizer, query=query, history=None)                            
    else:        
        response, history = self.chat(tokenizer, ins, history=input_history)                            
    
    temp_history = []
    for item in history:
        temp_history.append(
            {"role":"user","content":item[0]},
        )
        temp_history.append(
            {"role":"assistant","content":item[1]},
        )
        
    output = json.dumps({"response":response,"history":temp_history},ensure_ascii=False)
    return [(output,{"metadata":{}})] 


def init_model(model_dir,infer_params:Dict[str,str]={},sys_conf:Dict[str,str]={}): 
    pretrained_model_dir = os.path.join(model_dir,"pretrained_model")
    adaptor_model_dir = model_dir
    is_adaptor_model = os.path.exists(pretrained_model_dir)
    
    if not is_adaptor_model:        
        pretrained_model_dir = model_dir
    

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir,trust_remote_code=True)    
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_dir,trust_remote_code=True,                                                
                                                bf16=True,
                                                device_map='auto'
                                                ).half().cuda()
    if is_adaptor_model:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adaptor_model_dir)
        
    model.eval()
    model.generation_config = GenerationConfig.from_pretrained(pretrained_model_dir, trust_remote_code=True)       
    import types
    model.stream_chat = types.MethodType(stream_chat, model)     
    model.get_meta = types.MethodType(get_meta, model)
    return (model,tokenizer)






