import os

import numpy as np
import json
from typing import List

from base64 import b64encode
from typing import List,Any,Tuple

import gradio as gr
from byzerllm.bark.generation import SAMPLE_RATE
from scipy.io.wavfile import write as write_wav
import requests

# select finetune_model_predict(array(feature)) as a
def request(sql:str,json_data:str)->str:
    url = 'http://127.0.0.1:9003/model/predict'
    data = {
        'sessionPerUser': 'true',
        'owner': 'william',
        'dataType': 'string',
        'sql': sql,
        'data': json_data
    }
    response = requests.post(url, data=data)
    if response.status_code != 200:
        raise Exception(response.text)
    return response.text


def voice_to_text(rate:int, t:np.ndarray)->str:
    json_data = json.dumps([
        {"rate":rate, "voice": t.tolist()}
    ])

    response = request('''
     select voice_to_text(array(feature)) as value
    ''',json_data)

    t = json.loads(response)
    t2 = json.loads(t[0]["value"][0])
    return t2[0]["predict"]

def text_to_voice(s:str)->np.ndarray:    
    
    json_data = json.dumps([
        {"instruction":s}
    ])
    response = request('''
     select text_to_voice(array(feature)) as value
    ''',json_data)
    
    t = json.loads(response)
    t2 = json.loads(t[0]["value"][0])    
    return np.array(t2[0]["predict"])

## s,history = state.history
def chat(s:str,history:List[Tuple[str,str]])->str:
    newhis = [{"query":item[0],"response":item[1]} for item in history]
    json_data = json.dumps([
        {"instruction":s,"history":newhis,"output":"NAN"}
    ])
    response = request('''
     select chat(array(feature)) as value
    ''',json_data)    
    t = json.loads(response)
    t2 = json.loads(t[0]["value"][0])
    return t2[0]["predict"]


class UserState:
    def __init__(self,history:List[Tuple[str,str]]=[],output_state:str="") -> None:
        self.history = history
        self.output_state = output_state

    def add_chat(self,query,response):
        self.history.append((query,response)) 
        if len(self.history) > 10:
            self.history = self.history[len(self.history)-10:]    

    def add_output(self,message):        
        self.output_state = f"{self.output_state}\n\n{message}" 

    def clear(self):
        self.history = []
        self.output_state = ""    
    

def talk(t:str,state:UserState) -> str:
    prompt = '''
    你是威廉学院(college William)的一名外教，名字叫 William。 你的任务是指导我英文，包括
    为我提供学习计划，解决困扰。      
    ''';    
    s = prompt + t
    print("with prompt:",s)  
    s = chat(s,history = state.history)
    state.add_chat(t,s)
    return s
  

def html_audio_autoplay(bytes: bytes) -> object:
    """Creates html object for autoplaying audio at gradio app.
    Args:
        bytes (bytes): audio bytes
    Returns:
        object: html object that provides audio autoplaying
    """
    b64 = b64encode(bytes).decode()
    html = f"""
    <audio controls autoplay>
    <source src="data:audio/wav;base64,{b64}" type="audio/wav">
    </audio>
    """
    return html


def convert_to_16_bit_wav(data):
    # Based on: https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.write.html
    warning = "Trying to convert audio automatically from {} to 16-bit int format."
    if data.dtype in [np.float64, np.float32, np.float16]:        
        data = data / np.abs(data).max()
        data = data * 32767
        data = data.astype(np.int16)
    elif data.dtype == np.int32:    
        data = data / 65538
        data = data.astype(np.int16)
    elif data.dtype == np.int16:
        pass
    elif data.dtype == np.uint16:        
        data = data - 32768
        data = data.astype(np.int16)
    elif data.dtype == np.uint8:        
        data = data * 257 - 32768
        data = data.astype(np.int16)
    else:
        raise ValueError(
            "Audio data cannot be converted automatically from "
            f"{data.dtype} to 16-bit int format."
        )
    return data

def main_note(audio,text,state: UserState):
    if audio is None:
        return "",state.output_state,state

    rate, y = audio        
    print("voice to text:")

    t = voice_to_text(rate,y)

    if len(t.strip()) == 0 :        
        return "",state.output_state,state

    if  t.strip()=="重新开始":
        state.clear()
        return "",state.output_state,state
    
    print("talk to chatglm6b:")
    s = talk(t + " " + text,state)    
    print("chatglm6b:",s)    
    print("text to voice")    
    message = f"你: {t}\n\n外教: {s}\n"
    
    m = text_to_voice(s)
    
    from scipy.io.wavfile import write as write_wav
    import io
    wav_file = io.BytesIO()        
    write_wav(wav_file, SAMPLE_RATE, convert_to_16_bit_wav(m))
    wav_file.seek(0)
    html = html_audio_autoplay(wav_file.getvalue())

    state.add_output(message)
    return html,state.output_state,state

state = gr.State(UserState())
demo = gr.Interface(
    fn=main_note,
    inputs = [gr.Audio(source="microphone"),gr.TextArea(lines=30, placeholder="message"),state],
    outputs= ["html",gr.TextArea(lines=30, placeholder="message"),state],
    examples=[
    ],    
    interpretation=None,
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0",debug=True)
