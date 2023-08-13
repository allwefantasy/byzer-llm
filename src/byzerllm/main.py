from stable_diffusion import init_model

model, _ = init_model("/home/byzerllm/models/stable-diffusion-v1-5")
data = model.stream_chat(
    tokenizer=None, prompt="plane destory", negative_prompt="plane"
)
for d in data:
    d
