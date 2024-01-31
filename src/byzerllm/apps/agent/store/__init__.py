
import pydantic
from typing import Any, Dict
from abc import ABC, abstractmethod


class Message(pydantic.BaseModel):
    id: str
    m: Dict[str, Any]
    sender: str
    receiver: str
    timestamp: float


class MessageStore(ABC):    
    @abstractmethod
    def put(self, message: Message):
        pass
    
    @abstractmethod
    def get(self, id: str):
        pass

    @abstractmethod
    def clear(self, id: str):
        pass
