import json
from typing import List, Dict, Any
from loguru import logger

class CustomSaasAPI:
    def __init__(self, infer_params: Dict[str, str]) -> None:
        self.log_file = infer_params.get("saas.log_file", "filelogger.json")
        self.meta = {
            "model_deploy_type": "saas",
            "backend": "filelogger",
            "support_stream": False,
            "model_name": "filelogger",
        }

    def get_meta(self):
        return [self.meta]

    def chat_oai(self, conversations: List[Dict[str, Any]], **kwargs):
        try:
            with open(self.log_file, 'a') as f:
                json.dump(conversations, f)
                f.write('\n')
            return [{"output": "Messages logged successfully", "metadata": {}}]
        except Exception as e:
            logger.error(f"Error logging messages: {e}")
            return [{"output": f"Error: {str(e)}", "metadata": {}}]

    def stream_chat_oai(self, conversations: List[Dict[str, Any]], **kwargs):
        return self.chat_oai(conversations, **kwargs)