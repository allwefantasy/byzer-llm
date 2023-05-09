from byzerllm.chatglm6b.tunning.infer import init_model,predict

model,tokenizer = init_model("/my8t/byzerllm/jobs/checkpoint-17000")
s = predict("如何使用Byzer加载一个csv文件",model,tokenizer)
print(s)