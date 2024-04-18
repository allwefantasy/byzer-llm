from byzerllm.utils.client import ByzerLLM,InferBackend,Templates
import ray
import pytest

model_path = "/home/winubuntu/models/deepseek-coder-1.3b-instruct"
model_name = "chat"

class TestByzerLLMVLLMDeploy(object):
    llm = None

    def setup_class(self):
        if ray.is_initialized():
            ray.shutdown()
        
        ray.init(address="auto",namespace="default")
        
        self.llm = ByzerLLM(verbose=True)

        if self.llm.is_model_exist(model_name):
            self.llm.undeploy(model_name)

        self.llm.setup_gpus_per_worker(1).setup_num_workers(1).setup_infer_backend(InferBackend.VLLM)
        self.llm.deploy(
            model_path=model_path,
            pretrained_model_type="custom/auto",
            udf_name=model_name,
            infer_params={}
        )
        self.llm.setup_default_model_name(model_name)        
        
        
    def teardown_class(self):
        if self.llm.is_model_exist(model_name):
            self.llm.undeploy(model_name)

    def test_template(self):
        import json
        meta = self.llm.get_meta(model=model_name)        
        assert meta.get("support_chat_template",False) == True
        m = self.llm.apply_chat_template(model_name,json.dumps([{
            "content":"You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.",
            "role":"system"        
        },
            {
            "content":"写个冒泡算法",
            "role":"user"
        }],ensure_ascii=False))
        print(m,flush=True)
    
    def test_chat_oai(self):
        self.llm.setup_template(model_name,Templates.deepseek_code_chat())
        t = self.llm.chat_oai([{
            "content":"You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.",
            "role":"system"        
        },
            {
            "content":"写个冒泡算法",
            "role":"user"
        }])
        print("======",t)

    def test_insert(self):
        self.llm.setup_template(model_name,Templates.deepseek_code_insertion())
        t = self.llm.chat_oai([{
            "content":'''
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[0]
    left = []
    right = []
<｜fim▁hole｜>
        if arr[i] < pivot:
            left.append(arr[i])
        else:
            right.append(arr[i])
    return quick_sort(left) + [pivot] + quick_sort(right)
''',
            "role":"system"        
        }])
        print("======",t)  

    def test_completion(self):
        self.llm.setup_template(model_name,Templates.deepseek_code_completion())
        t = self.llm.chat_oai([{
            "content":'''
#write a quick sort algorithm
''',
            "role":"system"        
        }])
        print("======",t)      
    



    
