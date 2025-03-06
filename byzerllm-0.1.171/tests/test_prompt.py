import pytest
import byzerllm

@pytest.fixture(scope="module")
def setup():
    byzerllm.connect_cluster()
    llm = byzerllm.ByzerLLM()
    model = "qianwen_chat"
    llm.setup_default_model_name(model)
    llm.setup_template(model,"auto")
    return llm

class Wow:
    def __init__(self,llm):
        self.llm = llm

    @byzerllm.prompt(lambda self:self.llm)
    def test(self,s:str)->str: 
        '''
        Hello, {{ s }}!
        '''

    @byzerllm.prompt()
    def test2(self,s:str)->str: 
        '''
        Hello, {{ s }}!
        '''    

@byzerllm.prompt(options={"llm_config":{}})
def test(s:str)->str: 
    '''
    Hello, {{ s }}!
    '''  

def test_prompt_class_method(setup):
    w = Wow(setup)
    assert w.test.prompt("你是谁") == "\nHello, 你是谁!"
    assert isinstance(w.test.run("你是谁"),str)

def test_prompt_class_method_no_llm(setup):    
    w = Wow(setup)
    assert w.test2.prompt("你是谁") == "\nHello, 你是谁!"
    
def test_prompt_standalone_func(setup):
    test.with_llm(setup)
    assert test.prompt("你是谁") == "\nHello, 你是谁!"    
    
def test_prompt_options(setup):
    w = Wow(setup)    
    result = w.test.options({"model":"kimi_8k_chat","llm_config":{"max_length":1000}}).run("你是谁")
    assert "MoonshotAI" in result
    
def test_prompt_with_llm(setup):
    w = Wow(setup)
    assert isinstance(w.test.with_llm(setup).run("你是谁"))