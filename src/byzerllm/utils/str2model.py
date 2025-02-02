from typing import Any, Type
from byzerllm.utils.client import code_utils
from byzerllm.utils.nontext import TagExtractor
import json

def to_model(result: str,model_class: Type[Any]):                
    if not isinstance(result, str):
        raise ValueError("The decorated function must return a string")
    try:  
        # quick path for json string  
        if result.startswith("```json") and result.endswith("```"):
            json_str = result[len("```json"):-len("```")]
        else:
            json_str = code_utils.extract_code(result)[-1][1]
        json_data = json.loads(json_str)            
    except json.JSONDecodeError as e:
        print(f"The returned string is not a valid JSON, e: {str(e)} string: {result}")            
        tag_extractor = TagExtractor(result)
        result = tag_extractor.extract()
        json_data = json.loads(result.content)    
    
    try:
        if isinstance(json_data, list):
            return [model_class(**item) for item in json_data]    
        return model_class(**json_data)
    except TypeError:
        raise TypeError("Unable to create model instance from the JSON data")
