import pydantic
from typing import Optional, List, Dict, Any
from enum import Enum

class Action(Enum):
    CONTINUE = "continue"
    STOP = "stop"    

class QueryRewriteResult(pydantic.BaseModel):
    message: Dict[str, Any]    
    action: Action
    extra_info: Dict[str, Any]    