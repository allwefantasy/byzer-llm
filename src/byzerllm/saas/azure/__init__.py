import time
from typing import List, Tuple, Dict, Any, Union
import azure.cognitiveservices.speech as speechsdk
import ray
from byzerllm.utils.types import BlockVLLMStreamServer, StreamOutputs, SingleOutput, SingleOutputMeta, BlockBinaryStreamServer
import threading
import asyncio
import traceback
import uuid
import os

class CustomSaasAPI:

    def __init__(self, infer_params: Dict[str, str]) -> None:
        self.subscription_key = infer_params["saas.subscription_key"]
        self.service_region = infer_params["saas.service_region"]
        self.voice_name = infer_params.get("saas.voice_name", "en-US-AriaNeural")

        self.speech_config = speechsdk.SpeechConfig(subscription=self.subscription_key, region=self.service_region)
        self.speech_config.speech_synthesis_voice_name = self.voice_name

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
        request_id = [None]
        
        speech_config = self.speech_config
        speech_config.speech_synthesis_voice_name = voice
        
        if not stream:            
            speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
            start_time = time.monotonic()
            request_id[0] = str(uuid.uuid4())
            result = speech_synthesizer.speak_text_async(ins).get()
            audio_data = result.audio_data
            time_cost = time.monotonic() - start_time           
            return [(audio_data, {"metadata": {
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
                file_config = speechsdk.audio.AudioOutputConfig(filename=chunk_size)
                speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=file_config)
                                
                request_id[0] = str(uuid.uuid4())
                try:
                    result = speech_synthesizer.speak_text_async(ins).get()
                    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                        with open(chunk_size, "rb") as audio_file:
                            while True:
                                chunk = audio_file.read(chunk_size)
                                if not chunk:
                                    break
                                ray.get(server.add_item.remote(request_id[0], StreamOutputs(outputs=[SingleOutput(text=chunk, metadata=SingleOutputMeta(
                                        input_tokens_count=0,
                                        generated_tokens_count=0,
                                    ))])
                                ))
                except:
                    traceback.print_exc()
                
                ray.get(server.mark_done.remote(request_id[0]))
                if os.path.exists(chunk_size):
                    os.remove(chunk_size)
                                        
            threading.Thread(target=writer, daemon=True).start()
                   
            time_count = 10 * 100
            while request_id[0] is None and time_count > 0:
                time.sleep(0.01)
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
            voice = last_message.get("voice", self.voice_name)
            chunk_size = last_message.get("chunk_size", None)
            return await self.text_to_speech(stream=stream,
                                             ins=ins,
                                             voice=voice,
                                             chunk_size=chunk_size)

        raise Exception("Invalid input")