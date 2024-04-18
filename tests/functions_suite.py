import ray
from byzerllm.utils.client import ByzerLLM
from byzerllm.utils.client import code_utils,message_utils
from typing import List,Union,Optional
import pydantic


class TestResult(pydantic.BaseModel):
    tpe: str = pydantic.Field(description="测试类别")
    code: int = pydantic.Field(description="测试结果，0为通过，1为不通过")

ray.init(address="auto",namespace="default",ignore_reinit_error=True)

result = []

def test_functions(chat_model_name:str,enable_default_sys_message=False):    
    llm = ByzerLLM()
    llm.setup_default_model_name(chat_model_name)
    llm.setup_default_emb_model_name("emb")
    llm.setup_template(chat_model_name,"auto") 
    llm.setup_extra_generation_params(chat_model_name,extra_generation_params={
        "temperature":0.01,"top_p":0.99
    })
    
    ### function calling test
    def run_code():  
        '''
        用户表达肯定的观点或者代码没有问题，请调用我
        '''              
        return 0        
        
    def ignore():  
        '''
        用户表达否定或者代码不符合特定规范，可以调用我
        '''              
        return 1    



    temp_conversation = {
        "role":"user",
        "content":"我觉得代码没有什么问题，可以运行。\n注意，你只需要判断调用哪个函数，并不需要解决问题。",
    }    

    prefix = "system" if enable_default_sys_message else "prompt"

    t = llm.chat_oai(conversations=message_utils.padding_messages_merge([temp_conversation]),
                            tools=[run_code,ignore],
                            execute_tool=True,enable_default_sys_message=enable_default_sys_message)
    if t[0].values and t[0].values[0] == 0:
        result.append(TestResult(tpe=f"function calling ({prefix})",code=0))
    else:
        result.append(TestResult(tpe=f"function calling ({prefix})",code=1))

    ### function impl test
    class TimeRange(pydantic.BaseModel):
        '''
        时间区间
        格式需要如下： yyyy-MM-dd
        '''  
        
        start: str = pydantic.Field(...,description="开始时间.时间格式为 yyyy-MM-dd")
        end: str = pydantic.Field(...,description="截止时间.时间格式为 yyyy-MM-dd")


    def calculate_time_range():
        '''
        计算时间区间，时间格式为 yyyy-MM-dd. 
        '''
        pass 
    
    def target_calculate_time_range():
        from datetime import datetime, timedelta
        # 获取去年的年份
        last_year = datetime.now().year - 1
        
        # 定义开始和结束月份
        start_month = 3
        end_month = 7
        
        # 计算开始日期和结束日期
        start_date = datetime(last_year, start_month, 1).strftime("%Y-%m-%d")
        end_date = (datetime(last_year, end_month, 1) + timedelta(days=31)).strftime("%Y-%m-%d")
        
        # 确保结束日期不超过实际月份的天数
        end_date = min(end_date, (datetime(last_year, end_month, 1) + timedelta(days=30)).strftime("%Y-%m-%d"))
        
        return {"start": start_date, "end": end_date}
        
    t = llm.chat_oai([{
        "content":"去年三月到七月",
        "role":"user"    
    }],impl_func=calculate_time_range,response_class=TimeRange,execute_impl_func=True,enable_default_sys_message=enable_default_sys_message)

    if t[0].value and t[0].value == TimeRange.model_validate(target_calculate_time_range()):
        result.append(TestResult(tpe=f"function impl ({prefix})",code=0))
    else:
        result.append(TestResult(tpe=f"function impl ({prefix})",code=1))

    ### respond with class test        

    class Story(pydantic.BaseModel):
        '''
        故事
        '''

        title: str = pydantic.Field(description="故事的标题")
        body: str = pydantic.Field(description="故事主体")



    t = llm.chat_oai([
    {
        "content":f'''请给我讲个故事，分成两个部分，一个标题，一个故事主体''',
        "role":"user"
    },
    ],response_class=Story,enable_default_sys_message=enable_default_sys_message)

    if t[0].value and t[0].value.title and t[0].value.body:
        result.append(TestResult(tpe=f"respond with class ({prefix})",code=0))
    else:
        result.append(TestResult(tpe=f"respond with class ({prefix})",code=1))
    
    ## final result
    r = "\n".join([m.model_dump_json() for m in result])
    t = llm.chat_oai(conversations=[
    {
        "content":f'''我有一段下面的 json 格式数据：

```json
{r}
```        

请对上面数据进行一个简单的统计分析，最后输出一个 markdown 格式的表格，表格列名为：模型，功能类型,是否通过。

其中模型名称固定为：{chat_model_name}。
''',
        "role":"user"
    }],   ) 
    print(t[0].output)
    return t[0].output

if __name__ == "__main__":
    ## get command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='function test')
    parser.add_argument('--enable_default_sys_message', type=bool, default=False,help='enable default sys message')
    parser.add_argument('--chat_model_name', type=str, default="chat",help='chat model name')
    args = parser.parse_args()    
    test_functions(chat_model_name=args.chat_model_name,enable_default_sys_message=args.enable_default_sys_message)
 