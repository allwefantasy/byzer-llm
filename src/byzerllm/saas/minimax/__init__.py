import logging
import requests
import json
import traceback
from enum import Enum
from dataclasses import dataclass

from typing import Optional, List, Dict, Union, Any,str,Any

import tenacity
from tenacity import (
    before_sleep_log,
    wait_exponential,
)

logger = logging.getLogger(__name__)

DEFAULT_BOT_SETTING = 'You are a helpful assistant. Think it over and answer the user question correctly.'


class MiniMaxError(Exception):
    def __init__(
            self,
            request_id=None,
            status_msg=None,
            status_code=None,
            http_body=None,
            http_status=None,
            json_body=None,
            headers=None,
    ):
        super(MiniMaxError, self).__init__(
            f"api return error, code: {status_code}, msg: {status_msg}"
        )

        if http_body and hasattr(http_body, "decode"):
            try:
                http_body = http_body.decode("utf-8")
            except BaseException:
                http_body = (
                    "<Could not decode body as utf-8. "
                    "Please contact us through our help center at https://api.minimax.chat>"
                )

        self._status_msg = status_msg
        self.http_body = http_body
        self.http_status = http_status
        self.json_body = json_body
        self.headers = headers or {}
        self.status_code = status_code
        self.request_id = self.headers.get("request-id", request_id)

     # saas/proprietary
    def get_meta(self):
        return [{
            "model_deploy_type": "saas",
            "backend":"saas"
        }]    

    def __str__(self):
        msg = self._status_msg or "<empty message>"
        if self.request_id is not None:
            return "Request {0}: {1}".format(self.request_id, msg)
        elif self.status_code is not None:
            return "API return error, code: {0}, msg: {1}".format(self.status_code, msg)
        else:
            return msg

    def __repr__(self):
        return "%s(message=%r, http_status=%r, request_id=%r)" % (
            self.__class__.__name__,
            self._status_msg,
            self.http_status,
            self.request_id,
        )


def _minimax_api_retry_if_need(exception):
    """
        look for details: https://api.minimax.chat/document/guides/chat-pro?id=64b79fa3e74cddc5215939f4

        1000: 未知错误
        1001: 超时
        1002: 触发RPM限流
        1004: 鉴权失败
        1008: 余额不足
        1013: 服务内部错误
        1027: 输出内容错误
        1039: 触发TPM限流
        2013: 输入格式信息不正常
    """
    if isinstance(exception, MiniMaxError):
        status_code = exception.status_code
        return (status_code == 1000
                or status_code == 1001
                or status_code == 1002
                or status_code == 1013
                or status_code == 1039)
    return False


class CustomSaasAPI:
    def __init__(self, infer_params: Dict[str, str]) -> None:
        self.api_key = infer_params["saas.api_key"]
        self.group_id = infer_params["saas.group_id"]
        self.model = infer_params.get("saas.model", "abab5.5-chat")
        self.api_url = infer_params.get("saas.api_url", "https://api.minimax.chat/v1/text/chatcompletion_pro")

    def stream_chat(
            self,
            tokenizer,
            ins: str,
            his: List[Dict[str,Any]],
            max_length: int = 4096,
            top_p: float = 0.7,
            temperature: float = 0.9,
            **kwargs
    ):
        glyph = kwargs.get('glyph', None)
        bot_settings = MiniMaxBotSettings()
        messages = MiniMaxMessages()

        for item in his:
            role, content = item['role'], item['content']
            if role == "system":
                bot_settings.append("Assistant", content)
                continue
            messages.append(content, role)

        if ins:
            messages.append(ins, MiniMaxMessageRole.USER)

        if bot_settings.is_empty():
            bot_settings.append("Assistant", DEFAULT_BOT_SETTING)

        payload = {
            "model": self.model,
            "messages": messages.to_list(),
            "tokens_to_generate": max_length,
            "temperature": temperature,
            "top_p": top_p,
            "sample_messages": [],
            "plugins": [],
            "bot_setting": bot_settings.to_list(),
            "reply_constraints": {
                "sender_type": "BOT",
                "sender_name": "Assistant"
            }
        }

        if glyph:
            payload['reply_constraints']['glyph'] = glyph

        print(f"【Byzer --> MiniMax({self.model})】: {payload}")

        content = None
        try:
            content = self.request_with_retry(payload)
        except Exception as e:
            traceback.print_exc()
            if content == "" or content is None:
                content = f"request minimax api failed: {e}"
        return [(content, "")]

    @tenacity.retry(
        reraise=True,
        retry=tenacity.retry_if_exception(_minimax_api_retry_if_need),
        stop=tenacity.stop_after_attempt(10),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def request_with_retry(self, payload):
        """Use tenacity to retry the completion call."""

        api_url = f"{self.api_url.removesuffix('/')}?GroupId={self.group_id}"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        response = requests.post(api_url, data=json.dumps(payload), headers=headers)
        if response.status_code == 200:
            res_data = json.loads(response.text)
            print(f"【MiniMax({self.model}) --> Byzer】: {res_data}")
            base_status_code = res_data['base_resp']['status_code']
            if base_status_code != 0:
                raise MiniMaxError(
                    request_id=res_data.get('id'),
                    status_code=base_status_code,
                    status_msg=res_data['base_resp']['status_msg'],
                    headers=headers,
                    http_status=response.status_code
                )
            content = res_data["reply"].strip()
            return content
        else:
            print(
                f"request failed with status code `{response.status_code}`, "
                f"headers: `{response.headers}`, "
                f"body: `{response.content!r}`"
            )
            raise MiniMaxError(headers=headers, http_status=response.status_code, http_body=response.text)


class MiniMaxMessageRole(str, Enum):
    """MiniMax Message role."""
    USER = "USER"
    BOT = "BOT"
    FUNCTION = "FUNCTION"


class MiniMaxBotSettings:
    """
    MiniMax BotSetting list of Chat model.
    Look for details: https://api.minimax.chat/document/guides/chat-pro
    """

    @dataclass
    class Setting:
        """
        MiniMax bot setting.
        """
        bot_name: str = ""
        content: str = ""

        def to_dict(self) -> dict:
            """
            Convert generic bot setting to dict.
            """
            return {
                "bot_name": self.bot_name,
                "content": self.content
            }

    def __init__(self) -> None:
        """
        Init MiniMaxBotSettings
        """
        self._bot_settings: List[MiniMaxBotSettings.Setting] = []

    def append(self, bot_name: str, content: Optional[str] = DEFAULT_BOT_SETTING) -> None:
        """
        append setting to settings_list
        """
        self._bot_settings.append(MiniMaxBotSettings.Setting(bot_name=bot_name, content=content))

    def to_list(self) -> List[Dict[str, Any]]:
        """
        convert bot settings to list
        """
        return [bot.to_dict() for bot in self._bot_settings]

    def is_empty(self) -> bool:
        """
        check settings is empty
        """
        return len(self._bot_settings) <= 0


class MiniMaxMessages:
    """
    MiniMax Message list of Chat model.
    Look for details: https://api.minimax.chat/document/guides/chat-pro
    """

    @dataclass
    class Message:
        """
        MiniMax Chat message.
        """
        sender_type: Union[str, MiniMaxMessageRole] = MiniMaxMessageRole.USER
        sender_name: str = ""
        text: Optional[str] = ""

        def to_dict(self) -> dict:
            """
            Convert generic message to MiniMax message dict.
            """
            sender_type = self.sender_type
            if isinstance(sender_type, str):
                sender_type = self._mapping_sender_type()
            return {
                "sender_type": sender_type.value,
                "sender_name": self.sender_name,
                "text": self.text
            }

        def _mapping_sender_type(self) -> MiniMaxMessageRole:
            if self.sender_type == "system" or self.sender_type == "assistant":
                return MiniMaxMessageRole.BOT
            if self.sender_type == "function":
                return MiniMaxMessageRole.FUNCTION
            return MiniMaxMessageRole.USER

    def __init__(self, bot_name_mapping: Optional[dict] = None) -> None:
        """
        Init MiniMaxMessages
        """
        self._msg_list: List[MiniMaxMessages.Message] = []
        self._bot_name_mapping: dict = {
            "USER": "User",
            "user": "User",
            "BOT": "Assistant",
            "system": "Assistant",
            "assistant": "Assistant"
        } if bot_name_mapping is None else bot_name_mapping

    def append(
            self,
            message: str,
            sender_type: Optional[Union[str, MiniMaxMessageRole]] = None,
            sender_name: Optional[str] = None,
    ) -> None:
        """
        append message to message_list
        """
        sender_type = sender_type if sender_type is not None else MiniMaxMessageRole.USER
        if sender_name is None:
            sender_name = self._bot_name_mapping.get(sender_type) if sender_type in self._bot_name_mapping else "User"
        msg = MiniMaxMessages.Message(
            sender_type=sender_type,
            sender_name=sender_name,
            text=message
        )
        self._msg_list.append(msg)

    def to_list(self) -> List[Dict[str, Any]]:
        """
        convert messages to list
        """
        return [msg.to_dict() for msg in self._msg_list]
