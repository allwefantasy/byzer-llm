import byzerllm
llm = byzerllm.get_single_llm("ark_v3_0324_chat",product_mode="lite")

@byzerllm.prompt()
def hello(name:str)->str:
    '''
    你好 {{ name }}
    '''    


print(hello.with_llm(llm).run("张三"))