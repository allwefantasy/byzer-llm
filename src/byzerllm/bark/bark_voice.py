import os
import numpy as np
from .generation import (
    GenerateModel,SAMPLE_RATE
)
from .api import VoiceGenerateAPI

ZH_SPEAKER = "v2/zh_speaker_3"
EN_SPEAKER = "v2/en_speaker_6"

silence = np.zeros(int(0.25 * SAMPLE_RATE)) 

class Inference:
    def __init__(self,model:GenerateModel) -> None:
        self.model = model
        self.api = VoiceGenerateAPI(model)
        self.model.preload_models()
        self.texts = []

    def add(self,text:str,prompt:str) -> None:
        self.texts.append((text,prompt))

    def generate(self,text:str,prompt:str) -> np.ndarray:
        audio_array = self.api.generate_audio(text, history_prompt=prompt)
        return np.concatenate([audio_array, silence.copy()])

    def batch_generate(self) ->  np.ndarray:
        # later, we will use Ray to parallelize the generation
        pieces = []
        for text,prompt in self.texts:
            audio_array = self.api.generate_audio(text, history_prompt=prompt)
            pieces += [audio_array, silence.copy()]
        return np.concatenate(pieces)    


def build_void_infer(model_dir:str,tokenizer_dir:str) -> Inference:
    model = GenerateModel(model_dir=model_dir,
    tokenizer_dir=tokenizer_dir)        
    infer = Inference(model)
    return infer        