# from wudao.api_request import executeEngineV2, getToken, queryTaskResult
# import datetime
# import time
# import uuid
# from urllib.parse import urlparse,urlencode
# import base64

# from typing import Union, List, Tuple, Optional, Dict

# class SparkDeskAPI:
#     def __init__(self,app_id:str, app_key:str,app_secret:str) -> None:
#         self.APPID = app_id
#         self.APIKey = app_key
#         self.APISecret = app_secret
#         gpt_url = "ws://spark-api.xf-yun.com/v1.1/chat"
#         self.host = urlparse(gpt_url).netloc
#         self.path = urlparse(gpt_url).path
#         self.gpt_url = gpt_url

#     def gen_params(appid, question):
#         """
#         通过appid和用户的提问来生成请参数
#         """
#         data = {
#             "header": {
#                 "app_id": appid,
#                 "uid": "1234"
#             },
#             "parameter": {
#                 "chat": {
#                     "domain": "general",
#                     "random_threshold": 0.5,
#                     "max_tokens": 2048,
#                     "auditing": "default"
#                 }
#             },
#             "payload": {
#                 "message": {
#                     "text": [
#                         {"role": "user", "content": question}
#                     ]
#                 }
#             }
#         }
#         return data    

#     def create_url(self):
#         # 生成RFC1123格式的时间戳
#         now = datetime.now()
#         date = format_date_time(mktime(now.timetuple()))

#         # 拼接字符串
#         signature_origin = "host: " + self.host + "\n"
#         signature_origin += "date: " + date + "\n"
#         signature_origin += "GET " + self.path + " HTTP/1.1"

#         # 进行hmac-sha256进行加密
#         signature_sha = hmac.new(self.APISecret.encode('utf-8'), signature_origin.encode('utf-8'),
#                                  digestmod=hashlib.sha256).digest()

#         signature_sha_base64 = base64.b64encode(signature_sha).decode(encoding='utf-8')

#         authorization_origin = f'api_key="{self.APIKey}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_sha_base64}"'

#         authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')

#         # 将请求的鉴权参数组合为字典
#         v = {
#             "authorization": authorization,
#             "date": date,
#             "host": self.host
#         }
#         # 拼接鉴权参数，生成url
#         url = self.gpt_url + '?' + urlencode(v)
#         # 此处打印出建立连接时候的url,参考本demo的时候可取消上方打印的注释，比对相同参数时生成的url与自己代码生成的url是否一致
#         return url   
    
#     def stream_chat(self,tokenizer,ins:str, his:List[Tuple[str,str]]=[],  
#         max_length:int=4096, 
#         top_p:float=0.95,
#         temperature:float=0.1):  
#         data = {
#                     "topP": top_p,
#                     "temperature": temperature,
#                     "lengthPenalty": 1,
#                     "numBeams": 1,
#                     "minGenLength": 50,
#                     "requestTaskNo": str(uuid.uuid4()),
#                     "samplingStrategy": "BeamSearchStrategy",
#                     "maxTokens": max_length,
#                     "prompt": ins
#                 }    
#         token = self.temp_token if self.temp_token  else self.get_token_or_refresh()
#         resp = executeEngineV2(self.ability_type, self.engine_type, token, data)
        
#         if resp["code"] != 200:
#            token = self.get_token_or_refresh()
#            resp = executeEngineV2(self.ability_type, self.engine_type, token, data)

#         while resp["code"] == 200 and resp['data']['taskStatus'] == 'PROCESSING':
#             print(resp)
#             taskOrderNo = resp['data']['taskOrderNo']
#             time.sleep(1)
#             resp = queryTaskResult(token, taskOrderNo)
#         print(resp)
#         v = [resp['data']['outputText'][0]]
#         return [(res,"") for res in v]




