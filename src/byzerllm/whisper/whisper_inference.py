import numpy as np
from typing import Union, List, Tuple, Optional, Dict


class Inference:
    def __init__(
        self,        
        model_dir: str,
        device_index:int = 0        
    ) -> None:
        from faster_whisper import WhisperModel
        self.model_dir = model_dir
        try:
            import ray
            import os
            print("ray.get_gpu_ids(): {}".format(ray.get_gpu_ids()))
            print("CUDA_VISIBLE_DEVICES: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
        except Exception:
            pass    
        print(f"load whisper model: {self.model_dir}s cuda:{device_index}")
        self.model = WhisperModel(self.model_dir, device="cuda",compute_type="float16",device_index=device_index)

    def __call__(self, rate:int,t:np.ndarray, initial_prompt:str="以下是普通话的句子")->List[str]:        
        from scipy.io.wavfile import write as write_wav
        import io

        if t.dtype == "int64":
            t = t.astype(np.int32)

        byte_file = io.BytesIO()        
        write_wav(byte_file, rate, t)
        segments, info = self.model.transcribe(byte_file, beam_size=5,
        initial_prompt=initial_prompt
        )
        return [segment.text for segment in segments]