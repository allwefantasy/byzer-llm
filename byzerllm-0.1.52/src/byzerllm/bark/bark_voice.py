from typing import List
import numpy as np
from .generation import GenerateModel, SAMPLE_RATE
from .api import VoiceGenerateAPI

ZH_SPEAKER = "v2/zh_speaker_3"
EN_SPEAKER = "v2/en_speaker_6"

silence = np.zeros(int(0.25 * SAMPLE_RATE))


def is_chinese(t: str) -> bool:
    _is_chinese = True
    l = t.split(" ")
    element_num_15 = len([s for s in l if len(s.strip()) < 15])
    if element_num_15 / len(l) > 0.9:
        _is_chinese = False
    return _is_chinese


def raw_sentence(t: str) -> List[str]:
    import re

    pattern = (
        r"[。；：?！.:;?!]"  # regular expression pattern to match the punctuation marks
    )
    # split the string based on the punctuation marks
    segments = [s for s in re.split(pattern, t) if len(s.strip()) > 0]
    return segments


def not_toolong(t: str) -> List[str]:
    # split the string by white space
    # check the length of every segment which is greater than 20, get the count
    # use the count to divide the length of the list
    l = t.split(" ")
    _is_chinese = is_chinese(t)

    if _is_chinese and len(t) > 30:
        import re

        pattern = r"[。；：，?！,.:;?!]"
        segments = [s for s in re.split(pattern, t) if len(s.strip()) > 0]
        return segments
    elif not _is_chinese and len(l) > 30:
        import re

        pattern = r"[。；：，?！,.:;?!]"
        segments = [s for s in re.split(pattern, t) if len(s.strip()) > 0]
        return segments
    else:
        return [t]


class Inference:
    def __init__(self, model: GenerateModel) -> None:
        self.model = model
        self.api = VoiceGenerateAPI(model)
        self.model.preload_models()
        self.texts = []

    def add(self, text: str, prompt: str) -> None:
        self.texts.append((text, prompt))

    def generate(self, text: str, prompt: str) -> np.ndarray:
        audio_array = self.api.generate_audio(text, history_prompt=prompt)
        return np.concatenate([audio_array, silence.copy()])

    def batch_generate(self) -> np.ndarray:
        # later, we will use Ray to parallelize the generation
        pieces = []
        for text, prompt in self.texts:
            audio_array = self.api.generate_audio(text, history_prompt=prompt)
            pieces += [audio_array, silence.copy()]
        return np.concatenate(pieces)

    def text_to_voice(self, t1: str) -> np.ndarray:
        from langdetect import detect as detect_language

        t = t1.replace("\n", "")
        segments = []
        for s in raw_sentence(t):
            ss = not_toolong(s)
            for sss in ss:
                if len(sss.strip()) > 0:
                    segments.append(sss)
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

            print(f"{speaker} will speek: {s}")
            temps.append((s, speaker))

        results = []
        for temp in temps:
            print(f"temp: {temp}")
            import time

            # 统计耗时
            # 记录开始时间
            start_time = time.time()
            result = self.generate(*temp)
            end_time = time.time()
            # 计算执行时间
            execution_time = end_time - start_time
            # 打印执行时间
            print("代码执行时间：", execution_time, "秒")
            results.append(result)

        return np.concatenate(results)


def build_void_infer(model_dir: str, tokenizer_dir: str) -> Inference:
    model = GenerateModel(model_dir=model_dir, tokenizer_dir=tokenizer_dir)
    infer = Inference(model)
    return infer
