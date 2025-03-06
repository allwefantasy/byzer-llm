import _thread as thread
import base64
import datetime
import hashlib
import hmac
import json
from urllib.parse import urlparse
import ssl
from datetime import datetime
from time import mktime
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time
from typing import List, Tuple,Dict,Any
from byzerllm.utils.langutil import asyncfy_with_semaphore
import queue

import websocket

reponse_queue = queue.Queue()

class SparkDeskAPIParams(object):
    # 初始化
    def __init__(self, APPID, APIKey, APISecret, gpt_url, DOMAIN):
        self.APPID = APPID
        self.APIKey = APIKey
        self.APISecret = APISecret
        self.host = urlparse(gpt_url).netloc
        self.path = urlparse(gpt_url).path
        self.gpt_url = gpt_url
        self.DOMAIN = DOMAIN

        # 生成url
    def create_url(self):
        # 生成RFC1123格式的时间戳
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))

        # 拼接字符串
        signature_origin = "host: " + self.host + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + self.path + " HTTP/1.1"

        # 进行hmac-sha256进行加密
        signature_sha = hmac.new(self.APISecret.encode('utf-8'), signature_origin.encode('utf-8'),
                                 digestmod=hashlib.sha256).digest()

        signature_sha_base64 = base64.b64encode(signature_sha).decode(encoding='utf-8')

        authorization_origin = f'api_key="{self.APIKey}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_sha_base64}"'

        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')

        # 将请求的鉴权参数组合为字典
        v = {
            "authorization": authorization,
            "date": date,
            "host": self.host
        }
        # 拼接鉴权参数，生成url
        url = self.gpt_url + '?' + urlencode(v)
        # 此处打印出建立连接时候的url,参考本demo的时候可取消上方打印的注释，比对相同参数时生成的url与自己代码生成的url是否一致
        return url


class CustomSaasAPI:

    def __init__(self, infer_params: Dict[str, str]) -> None:
        required_params = [ "saas.appid", "saas.api_key", "saas.api_secret"]
        for param in required_params:
            if list(infer_params.keys()).count(param) < 1:
                raise ValueError("The parameter %s is a required field, please configure it"% param)
        for value in self.get_value(infer_params,required_params):
            if value is None or value == "":
                raise ValueError("The mandatory model parameters cannot be empty.")
        self.appid: str = infer_params["saas.appid"]
        self.api_key: str = infer_params["saas.api_key"]
        self.api_secret: str = infer_params["saas.api_secret"]
        self.gpt_url: str = infer_params.get("saas.gpt_url","wss://spark-api.xf-yun.com/v3.1/chat")
        self.domain: str = infer_params.get("saas.domain","generalv3")
        self.config = SparkDeskAPIParams(self.appid, self.api_key, self.api_secret, self.gpt_url, self.domain)
        self.timeout = int(infer_params.get("saas.timeout",30))
        self.debug = infer_params.get("saas.debug",False)

    @staticmethod
    def on_error(ws, error):
        pass


    @staticmethod
    def on_close(ws,a,b):
        pass


    @staticmethod
    def on_open(ws):
        thread.start_new_thread(CustomSaasAPI.run, (ws,))

    @staticmethod
    def run(ws, *args):
        # 8192
        data = {
            "header": {
                "app_id": ws.appid,
                "uid": "1234"
            },
            "parameter": {
                "chat": {
                    "domain": ws.domain,
                    "random_threshold": ws.temperature,
                    "max_tokens": ws.max_length,
                    "auditing": "default"
                }
            },
            "payload": {
                "message": {
                    "text": ws.question
                }
            }
        }
        data = json.dumps(data)        
        ws.send(data)


    @staticmethod
    def on_message(ws, message):
        data = json.loads(message)        
        code = data['header']['code']
        if code != 0:
            reponse_queue.put(f'请求错误: {code}, {data}')
            reponse_queue.put(None)
            ws.close()
        else:
            choices = data["payload"]["choices"]
            status = choices["status"]
            content = choices["text"][0]["content"]
            reponse_queue.put(content)
            if status == 2:
                reponse_queue.put(None)
                ws.close()


    # saas/proprietary
    def get_meta(self):
        return [{
            "model_deploy_type": "saas",
            "backend":"saas"
        }]
    
    async def async_get_meta(self):
        return await asyncfy_with_semaphore(self.get_meta)()

    def get_value(self,infer_params: Dict[str, str],keys_to_get):
        values = []
        for key in keys_to_get:
            if key in infer_params.keys():
                values.append(infer_params[key])
        return values
    
    async def async_stream_chat(self,tokenizer,ins:str, his:List[Dict[str,Any]]=[],
                    max_length:int=4096,
                    top_p:float=0.7,
                    temperature:float=0.9):
        return await asyncfy_with_semaphore(self.stream_chat)(tokenizer,ins,his,max_length,top_p,temperature)

    def stream_chat(self,tokenizer,ins:str, his:List[Dict[str,Any]]=[],
                    max_length:int=4096,
                    top_p:float=0.7,
                    temperature:float=0.9):

        q = his + [{"role": "user", "content": ins}]
        websocket.enableTrace(self.debug)
        wsUrl = self.config.create_url()
        ws = websocket.WebSocketApp(wsUrl,
                                    on_message=CustomSaasAPI.on_message,
                                    on_error=CustomSaasAPI.on_error,
                                    on_close=CustomSaasAPI.on_close,
                                    on_open=CustomSaasAPI.on_open)
        ws.appid = self.config.APPID
        ws.domain = self.config.DOMAIN
        ws.question = q
        ws.max_length = max_length
        ws.top_p = top_p
        ws.temperature = temperature
        ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})

        result = []

        t  = reponse_queue.get(timeout=self.timeout)
        while t is not None:
            result.append(t)
            t  = reponse_queue.get(timeout=self.timeout)       
         
        return [("".join(result),"")]