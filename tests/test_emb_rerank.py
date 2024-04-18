import importlib

from byzerllm.utils.client import ByzerLLM, InferBackend
import ray
import pytest

class TestByzerEmbRerank(object):
    llm = None
    udf_name = "emb_rerank"

    def setup_class(self):
        if ray.is_initialized():
            ray.shutdown()

        ray.init(address="auto", namespace="default")

        self.llm = ByzerLLM(verbose=True)

        if self.llm.is_model_exist(self.udf_name):
            self.llm.undeploy(self.udf_name)

        self.llm.setup_gpus_per_worker(0).setup_num_workers(1).setup_infer_backend(InferBackend.Transformers)

        self.llm.deploy(
            model_path="/Users/wanghan/data/bge-reranker-base",
            pretrained_model_type="custom/bge_rerank",
            udf_name=self.udf_name,
            infer_params={}
        )
        self.llm.setup_default_emb_model_name(self.udf_name)

    def teardown_class(self):
        if self.llm.is_model_exist(self.udf_name):
            self.llm.undeploy(self.udf_name)

    def test_rerank_local(self):
        infer_module = importlib.import_module('byzerllm.bge_rerank')
        model = infer_module.init_model("/Users/wanghan/data/bge-reranker-base", infer_params={}, sys_conf={})[0]
        score = model.embed_rerank(sentence_pairs=['query', 'passage'])
        print(score)

        scores = model.embed_rerank(sentence_pairs=[['what is panda?', 'hi'], ['what is panda?','The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']])
        print(scores)

    def test_rerank_remote(self):
        sentence_pairs_01 = ['query', 'passage']
        sentence_pairs_02 = [['what is panda?', 'hi'], ['what is panda?','The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']]

        t1 = self.llm.emb_rerank(sentence_pairs=sentence_pairs_01)
        print(t1[0].output, flush=True)
        t2 = self.llm.emb_rerank(sentence_pairs=sentence_pairs_02)
        print(t2[0].output, flush=True)

