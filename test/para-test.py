import ray
import os

import numpy as np
import json
from typing import List

from base64 import b64encode
from typing import List,Any,Tuple
from byzerllm.bark.generation import SAMPLE_RATE
import requests


ray.util.connect(conn_str="127.0.0.1:10001")

def request(sql:str,json_data:str)->str:
    url = 'http://127.0.0.1:9003/model/predict'
    data = {
        'sessionPerUser': 'true',
        'sessionPerRequest': 'true',
        'owner': 'william',
        'dataType': 'string',
        'sql': sql,
        'data': json_data
    }
    response = requests.post(url, data=data)
    if response.status_code != 200:
        raise Exception(response.text)
    return response.text

def chat(s:str,history:List[Tuple[str,str]])->str:
    newhis = [{"query":item[0],"response":item[1]} for item in history]
    json_data = json.dumps([
        {"instruction":s,"history":newhis,"output":"NAN"}
    ])
    response = request('''
     select chat(array(feature)) as value
    ''',json_data)    
    t = json.loads(response)
    t2 = json.loads(t[0]["value"][0])
    return t2[0]["predict"]  

@ray.remote
def new_chat(s:str):
    return chat(s,[])


refs = [new_chat.remote("你好") for i in range(10)]

for ref in refs:
    print(ray.get(ref))