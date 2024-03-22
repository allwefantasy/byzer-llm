from byzerllm.apps.agent import Agents
from byzerllm.apps.agent.user_proxy_agent import UserProxyAgent
from byzerllm.apps.agent.extensions.data_analysis import DataAnalysis
from byzerllm.utils.client import ByzerLLM,Templates
import os

chat_model_name = "qianwen_chat"
llm = ByzerLLM()
llm.setup_default_model_name(chat_model_name) 
llm.setup_template(chat_model_name,template=Templates.qwen())

current_file_directory = os.path.dirname(os.path.abspath(__file__))


def test_get_agent_names():
    data_analysis = DataAnalysis("chat4","william",os.path.join(current_file_directory,"test.csv"),
                             llm,None,skip_preview_file=True)
    
    v = data_analysis.get_agent_names()
    assert set(v) == set(['assistant_agent', 'visualization_agent', 'common_agent', 'privew_file_agent', 'python_interpreter'])
    data_analysis.close()

def test_update_pipeline_system_message():    
    data_analysis = DataAnalysis("chat4","william",os.path.join(current_file_directory,"test.csv"),
                             llm,None,skip_preview_file=True)
    
    data_analysis.update_pipeline_system_message("hello")
    assert data_analysis.get_pipeline_system_message() == "hello"
    data_analysis.close()

def test_system_message_by_agent():
    data_analysis = DataAnalysis("chat4","william",os.path.join(current_file_directory,"test.csv"),
                             llm,None,skip_preview_file=True)
    
    data_analysis.update_agent_system_message("assistant_agent","hello")
    assert data_analysis.get_agent_system_message("assistant_agent") == "hello"
    data_analysis.close()