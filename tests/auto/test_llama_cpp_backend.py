import os
import pytest
from unittest.mock import MagicMock, patch

from llama_cpp import Llama
from byzerllm.auto.backend_llama_cpp import LlamaCppBackend


@pytest.fixture(scope="module")
def llama_cpp_model():
    model_dir = "path/to/model/dir"
    infer_params = {}
    sys_conf = {}
    with patch.object(Llama, "__init__", lambda self, model_path: None):
        model = Llama(model_path="dummy_path")
        model.get_meta = MagicMock(return_value=[{"message_format": True}])
        model.chat = MagicMock(return_value="Generated text")
        model.n_ctx = 2048
        return LlamaCppBackend.init_model(model_dir, infer_params, sys_conf)


def test_init_model(llama_cpp_model):
    model, _ = llama_cpp_model
    assert isinstance(model, Llama)


def test_generate_with_chat(llama_cpp_model):
    model, _ = llama_cpp_model
    tokenizer = MagicMock()
    tokenizer.return_value = {"input_ids": [1, 2, 3]}

    result = LlamaCppBackend.generate(model, tokenizer, "Test input", his=[], max_length=100)

    assert len(result) == 1
    generated_text, metadata = result[0]
    assert generated_text == "Generated text"
    assert metadata["metadata"]["input_tokens_count"] == -1
    assert metadata["metadata"]["generated_tokens_count"] == 3
    assert metadata["metadata"]["prob"] == -1.0


@patch("asyncio.create_task")
@patch("ray.get_actor")
def test_generate_with_stream(mock_get_actor, mock_create_task, llama_cpp_model):
    model, _ = llama_cpp_model
    model.get_meta.return_value = [{"message_format": False}]
    model.return_value = {"choices": [{"text": "Generated text"}]}
    tokenizer = MagicMock()
    tokenizer.return_value = {"input_ids": [1, 2, 3]}

    mock_server = MagicMock()
    mock_get_actor.return_value = mock_server

    result = LlamaCppBackend.generate(model, tokenizer, "Test input", stream=True, request_id="test_id")

    assert len(result) == 1
    assert result[0][0] == ""
    assert result[0][1]["metadata"]["request_id"] == "test_id"
    assert result[0][1]["metadata"]["stream_server"] == "VLLM_STREAM_SERVER"
    mock_create_task.assert_called_once()
    mock_server.add_item.remote.assert_called_with("test_id", "RUNNING")


def test_get_meta(llama_cpp_model):
    model, _ = llama_cpp_model

    meta = LlamaCppBackend.get_meta(model)

    assert len(meta) == 1
    assert meta[0]["model_deploy_type"] == "proprietary"
    assert meta[0]["backend"] == "llama.cpp"
    assert meta[0]["max_model_len"] == 2048
    assert meta[0]["architectures"] == ["LlamaCpp"]
    assert meta[0]["message_format"] == True