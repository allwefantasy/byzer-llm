from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

model_dir =  "/home/winubuntu/projects/falcon-7b-instruct"

model = AutoModelForCausalLM.from_pretrained(model_dir,trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_dir)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)
sequences = pipeline(
   "请问如何才能吃得更加营养",
    max_length=200,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
)
print("begin")
for seq in sequences:
    print(f"Result: {seq['generated_text']}")

print("end")
import time
while True:
    time.sleep(3)