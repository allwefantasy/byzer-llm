from math import log2, pow
import os

import numpy as np
from scipy.fftpack import fft
from typing import List

from base64 import b64encode

import gradio as gr
from byzerllm.bark.bark_voice import Inference, build_void_infer, ZH_SPEAKER, EN_SPEAKER
from byzerllm.bark.generation import SAMPLE_RATE
from scipy.io.wavfile import write as write_wav
from faster_whisper import WhisperModel
from byzerllm.chatglm6b.tunning.infer import init_model as init_chatbot_model,predict as chat
from byzerllm.bark.generation import SAMPLE_RATE



text_to_voice_m = build_void_infer(
    model_dir="/home/winubuntu/projects/bark-model",
    tokenizer_dir="/home/winubuntu/projects/bert-base-multilingual-cased")

voice_to_text_m = WhisperModel("/home/winubuntu/projects/whisper-models/faster-whisper-large-v2", 
device="cuda",device_index=1, compute_type="float16")

chatbot_m,chatbot_t = init_chatbot_model("/my8t/byzerllm/jobs/checkpoint-17000/pretrained_model")

class UserState:
    def __init__(self,history=[],output_state:str="") -> None:
        self.history = history
        self.output_state = output_state

    def add_prompt(self,prompt):
        self.history.append(prompt) 

        if len(self.history) > 10:
            self.history = self.history[len(self.history)-10:]

    def get_prompt(self):
        return " ".join(self.history)

    def add_output(self,message):        
        self.output_state = f"{self.output_state}\n\n{message}" 

def voice_to_text(rate,t:np.ndarray)->List[str]:

    from scipy.io.wavfile import write as write_wav
    import io
    byte_file = io.BytesIO()
    write_wav(byte_file, rate, t)

    segments, info = voice_to_text_m.transcribe(byte_file, beam_size=5,
      initial_prompt="以下是普通话的句子"
    )
    return [segment.text for segment in segments]
    

def talk(t:str,state:UserState) -> str:
    prompt = '''
    你是威廉学院的一名外教，名字叫 William。 你的任务是指导我英文，包括
    为我提供学习计划，解决困扰。
    
    下面内容不要直接对我说，但是你需要自己记住，当我咨询你翻译的时候，你需要先用中文回答说好的。然后再说翻译的内容。
    当你翻译完成后，你需要重新说中文。
    
    ''';
    state.add_prompt(t)    
    s = chat(prompt + state.get_prompt(),chatbot_m,chatbot_t)
    return s

def is_chinese(t:str) -> bool:
    _is_chinese = True
    l = t.split(" ")
    element_num_15 = len([s for s in l if len(s.strip()) < 15])
    if element_num_15/len(l) > 0.9:
        _is_chinese = False
    return _is_chinese    

def raw_sentence(t:str) -> List[str]:
    import re       
    pattern = r'[。；：?！.:;?!]'  # regular expression pattern to match the punctuation marks
    # split the string based on the punctuation marks
    segments = [s for s in re.split(pattern, t) if len(s.strip()) > 0]
    return segments

def not_toolong(t:str) -> List[str]: 
    # split the string by white space 
    # check the length of every segment which is greater than 20, get the count
    # use the count to divide the length of the list
    l = t.split(" ")        
    _is_chinese = is_chinese(t)
    
    if(_is_chinese and len(t) > 30):
        import re           
        pattern = r'[。；：，?！,.:;?!]'         
        segments = [s for s in re.split(pattern, t) if len(s.strip()) > 0]
        return segments
    elif(not _is_chinese and len(l) > 30):
        import re           
        pattern = r'[。；：，?！,.:;?!]'         
        segments = [s for s in re.split(pattern, t) if len(s.strip()) > 0]
        return segments    
    else:
        return [t]   

def text_to_voice(t:str) -> np.ndarray:    
    from langdetect import detect as detect_language     
    segments = []
    for s in raw_sentence(t):
        ss = not_toolong(s)
        for sss in ss:
            if len(sss.strip()) > 0:
                segments.append(sss)
    print("sentences to:",segments)
    temps = []

    for s in segments: 
      lang = ""       
      try:
        lang = detect_language(s)
      except Exception:
        pass

      speaker = ZH_SPEAKER
      if lang == "en":
          speaker = EN_SPEAKER

      temps.append(text_to_voice_m.generate(s, speaker))
    return np.concatenate(temps)    

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

def main_note(audio,state: UserState):
    if audio is None:
        return None,"",state.output_state,state

    rate, y = audio        
    print("voice to text:")

    t = " ".join(voice_to_text(rate,y))

    if len(t.strip()) == 0:
        return None,"",state.output_state,state

    print("text:",t)
    print("talk to chatglm6b:")
    s = talk(t,state)    
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
    return (SAMPLE_RATE,m),html,state.output_state,state

state = gr.State(UserState())
demo = gr.Interface(
    fn=main_note,
    inputs = [gr.Audio(source="microphone"),state],
    outputs= [gr.Audio(),"html",gr.TextArea(lines=30, placeholder="message"),state],
    examples=[
        [os.path.join(os.path.abspath(''),"audio/recording1.wav")],
        [os.path.join(os.path.abspath(''),"audio/cantina.wav")],
    ],
    live=True,
    interpretation=None,
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0",debug=True)
