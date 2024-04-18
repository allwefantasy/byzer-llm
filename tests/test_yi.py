from byzerllm.utils.client import ByzerLLM,InferBackend,Templates
import ray
import pytest

model_path = "/home/winubuntu/models/Yi-6B-Chat-4bits"
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
            infer_params={"backend.quantization":"AWQ"}
        )
        self.llm.setup_default_model_name(model_name)        
        
        
    def teardown_class(self):
        if self.llm.is_model_exist(model_name):
            self.llm.undeploy(model_name)
    
    def test_chat_oai(self):
        self.llm.setup_template(model_name,Templates.yi())
        t = self.llm.chat_oai([{
            "content":"You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.",
            "role":"system"        
        },
            {
            "content":"写个冒泡算法",
            "role":"user"
        }])
        print("======",t)

       
    



    
