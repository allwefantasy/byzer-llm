from byzerllm.utils.client import ByzerLLM
import pytest

def test_byzerllm():
    llm = ByzerLLM()
    assert llm.verbose == False
   