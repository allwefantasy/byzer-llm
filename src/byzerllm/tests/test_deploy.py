from byzerllm.utils.client import ByzerLLM,InferBackend,Templates
import ray
import pytest

model_path = "/home/winubuntu/models/Qwen-1_8B-Chat"

class TestByzerLLMVLLMDeploy(object):
    llm = None

    def setup_class(self):
        if ray.is_initialized():
            ray.shutdown()
        
        ray.init(address="auto",namespace="default")
        
        self.llm = ByzerLLM(verbose=True)

        if self.llm.is_model_exist("chat"):
            self.llm.undeploy("chat")

        self.llm.setup_gpus_per_worker(1).setup_num_workers(1).setup_infer_backend(InferBackend.VLLM)
        self.llm.deploy(
            model_path=model_path,
            pretrained_model_type="custom/auto",
            udf_name="chat",
            infer_params={"backend.max_num_batched_tokens":8192,"backend.max_model_len":8192}
        )
        self.llm.setup_default_model_name("chat")
        self.llm.setup_template("chat",Templates.qwen())
        
        
    def teardown_class(self):
        if self.llm.is_model_exist("chat"):
            self.llm.undeploy("chat")
    
    def test_chat_oai(self):
        t = self.llm.chat_oai([{
            "content":"你好",
            "role":"user"
        }])
        assert len(t[0].output) > 1





    
