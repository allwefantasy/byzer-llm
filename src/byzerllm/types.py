from pydantic import BaseModel

class Bool(BaseModel):
    value: bool

class Int(BaseModel):
    value: int

class Float(BaseModel):
    value: float    

