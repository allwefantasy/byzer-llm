from faster_whisper import WhisperModel

# Run on GPU with FP16
model = WhisperModel("/home/winubuntu/projects/whisper-models/faster-whisper-large-v2", device="cuda", compute_type="float16")

# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
# model = WhisperModel(model_size, device="cpu", compute_type="int8")

segments, info = model.transcribe(
    "/home/winubuntu/Downloads/byzer-llm-introduction.wav", beam_size=5,
    initial_prompt="以下是普通话的句子,里面 白色 请生成 Byzer"
    )

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))