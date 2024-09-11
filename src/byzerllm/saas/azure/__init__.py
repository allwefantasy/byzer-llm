import time
from typing import List, Tuple, Dict, Any, Union
import ray
from byzerllm.utils.types import BlockVLLMStreamServer, StreamOutputs, SingleOutput, SingleOutputMeta, BlockBinaryStreamServer
import threading
import asyncio
import traceback
import uuid
import os
import json
import base64
from byzerllm.utils.langutil import asyncfy_with_semaphore

try:
    import azure.cognitiveservices.speech as speechsdk
except ImportError:
    raise ImportError("""
    Importing the Speech SDK for Python failed.
    Refer to
    https://docs.microsoft.com/azure/cognitive-services/speech-service/quickstart-text-to-speech-python for
    installation instructions.
    """)        

class CustomSaasAPI:

    def __init__(self, infer_params: Dict[str, str]) -> None:
        self.api_key = infer_params["saas.api_key"]
        self.service_region = infer_params.get("saas.service_region","eastus")
        self.base_url = infer_params.get("saas.base_url", None)
        
        self.speech_config = speechsdk.SpeechConfig(subscription=self.api_key, region=self.service_region)
        if  self.base_url is not None:
            self.speech_config.endpoint_id = self.base_url
        
        self.max_retries = int(infer_params.get("saas.max_retries", 10))

        self.meta = {
            "model_deploy_type": "saas",
            "backend": "saas",
            "support_stream": True,
            "model_name": "azure_tts",
        }

        try:
            ray.get_actor("BLOCK_VLLM_STREAM_SERVER")
        except ValueError:
            try:
                ray.remote(BlockVLLMStreamServer).options(name="BLOCK_VLLM_STREAM_SERVER", lifetime="detached", max_concurrency=1000).remote()
            except Exception as e:
                pass
        try:
            ray.get_actor("BlockBinaryStreamServer")
        except ValueError:
            try:
                ray.remote(BlockBinaryStreamServer).options(name="BlockBinaryStreamServer", lifetime="detached", max_concurrency=1000).remote()
            except Exception as e:
                pass

    def get_meta(self):
        return [self.meta]
    
    async def async_get_meta(self):
        return await asyncfy_with_semaphore(self.get_meta)()

    def process_input(self, ins: Union[str, List[Dict[str, Any]], Dict[str, Any]]):
        if isinstance(ins, list) or isinstance(ins, dict):
            return ins

        content = []
        try:
            ins_json = json.loads(ins)
        except:
            return ins

        if isinstance(ins_json, dict):
            return ins_json
       
        content = []
        for item in ins_json:
            if "image" in item or "image_url" in item:
                image_data = item.get("image", item.get("image_url", ""))
                if not image_data.startswith("data:"):
                    image_data = "data:image/jpeg;base64," + image_data
                content.append({"image_url": {"url": image_data}, "type": "image_url",})
            elif "text" in item:
                text_data = item["text"]
                content.append({"text": text_data, "type": "text"})
        if not content:
            return ins

        return content 

    async def text_to_speech(self, stream: bool, ins: str, voice: str, chunk_size: int = None,  **kwargs):
        response_format = kwargs.get("response_format", "mp3")
        language = kwargs.get("language", "zh-CN")
        
        request_id = [None]
        
        speech_config = self.speech_config
        speech_config.speech_synthesis_voice_name = voice or "zh-CN-XiaoxiaoNeural"
        speech_config.speech_synthesis_language = language
        
        format = speechsdk.SpeechSynthesisOutputFormat.Audio48Khz192KBitRateMonoMp3
        if response_format == "wav":
            format = speechsdk.SpeechSynthesisOutputFormat.Riff24Khz16BitMonoPcm            

        speech_config.set_speech_synthesis_output_format(format)
        
        if not stream:                       
            speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
            start_time = time.monotonic()
            request_id[0] = str(uuid.uuid4())
            result = speech_synthesizer.speak_text_async(ins).get()                         

            if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:                                        
                if result.cancellation_details.reason == speechsdk.CancellationReason.Error:
                    if result.cancellation_details.error_details:
                        raise Exception("Error details: {}".format(result.cancellation_details.error_details))
                       
            audio_data = result.audio_data            
            print(len(audio_data),flush=True)
            base64_audio = base64.b64encode(audio_data).decode()
            time_cost = time.monotonic() - start_time  
            del result
            del speech_synthesizer         
            return [(base64_audio, {"metadata": {
                "request_id": "",
                "input_tokens_count": 0,
                "generated_tokens_count": 0,
                "time_cost": time_cost,
                "first_token_time": 0,
                "speed": 0,
            }})]
        else:
            server = ray.get_actor("BlockBinaryStreamServer")
            
            def writer():
                request_id[0] = str(uuid.uuid4())
                pull_stream = speechsdk.audio.PullAudioOutputStream()            
                stream_config = speechsdk.audio.AudioOutputConfig(stream=pull_stream) 
                speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=stream_config)                                
                
                try:                                        
                    result = speech_synthesizer.speak_text_async(ins).get()                    
                    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                        del result
                        del speech_synthesizer                         
                        audio_buffer = bytes(32000)                          
                        filled_size = pull_stream.read(audio_buffer)
                        while filled_size > 0:                                                                                                                                    
                            ray.get(server.add_item.remote(request_id[0], 
                                                           StreamOutputs(outputs=[SingleOutput(text=audio_buffer[0:filled_size], 
                                                                                                              metadata=SingleOutputMeta(
                                        input_tokens_count=0,
                                        generated_tokens_count=0,
                                    ))])
                                ))
                            filled_size = pull_stream.read(audio_buffer)                                                                                                    
                    else:
                        raise Exception(f"Failed to synthesize audio: {result.reason}")                                       
                except:
                    traceback.print_exc()                    
                                
                ray.get(server.mark_done.remote(request_id[0]))                               
                                        
            threading.Thread(target=writer, daemon=True).start()
                   
            time_count = 10 * 100
            while request_id[0] is None and time_count > 0:
                await asyncio.sleep(0.01)
                time_count -= 1

            if request_id[0] is None:
                raise Exception("Failed to get request id")

            def write_running():
                return ray.get(server.add_item.remote(request_id[0], "RUNNING"))

            await asyncio.to_thread(write_running)
            return [("", {"metadata": {"request_id": request_id[0], "stream_server": "BlockBinaryStreamServer"}})]
            

    def speech_to_text(self, ins: str, **kwargs):
        return None

    def image_to_text(self, ins: str, **kwargs):
        return None

    def text_to_image(self, ins: str, **kwargs):
        return None

    def text_to_text(self, ins: str, **kwargs):
        return None

    async def async_stream_chat(self, tokenizer, ins: str, his: List[Dict[str, Any]] = [],
                                max_length: int = 4096,
                                top_p: float = 0.7,
                                temperature: float = 0.9, **kwargs):

        stream = kwargs.get("stream", False)

        messages = [{"role": message["role"], "content": self.process_input(message["content"])} for message in
                    his] + [{"role": "user", "content": self.process_input(ins)}]
        last_message = messages[-1]["content"]

        if isinstance(last_message, dict) and "input" in last_message:
            voice = last_message.get("voice", "zh-CN-XiaoxiaoNeural")
            chunk_size = last_message.get("chunk_size", None)
            input = last_message["input"]
            response_format = last_message.get("response_format", "mp3")
            language = last_message.get("language","zh-CN")
            return await self.text_to_speech(stream=stream,
                                             ins=input,
                                             voice=voice,
                                             chunk_size=chunk_size,
                                             response_format=response_format,language=language
                                             )

        raise Exception("Invalid input")