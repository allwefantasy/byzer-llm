from typing import List, Dict,Any
import copy

def termindate_message(message:Dict[str,Any]):
    if "metadata" not in message:
        message["metadata"] = {}
    message["metadata"]["TERMINATE"] = True
    return message

def un_termindate_message(message:Dict[str,Any]):
    if "metadata" not in message:
        message["metadata"] = {}
    message["metadata"]["TERMINATE"] = False
    return message

def success_message(message:Dict[str,Any]):
    if "metadata" not in message:
        message["metadata"] = {}
    message["metadata"]["code"] = 0
    return message

def fail_message(message:Dict[str,Any]):
    if "metadata" not in message:
        message["metadata"] = {}
    message["metadata"]["code"] = 1
    return message

def is_success(message:Dict[str,Any]):
    if "metadata" not in message or "code" not in message["metadata"]:
        return False
    return message["metadata"]["code"] == 0

def copy_error_count(message:Dict[str,Any],new_message:Dict[str,Any]):
    if "metadata" not in message:
        message["metadata"] = {}
    if "metadata" not in new_message:
        new_message["metadata"] = {}
    new_message["metadata"]["error_count"] = message["metadata"].get("error_count",0)
    return new_message

def get_error_count(message:Dict[str,Any]):
    if "metadata" not in message:
        message["metadata"] = {}   
    return message["metadata"].get("error_count",0)

def inc_error_count(message:Dict[str,Any]):
    if "metadata" not in message:
        message["metadata"] = {}
    message["metadata"]["error_count"] = message["metadata"].get("error_count",0) + 1
    return message

def check_error_count(message:Dict[str,Any],max_error_count:int=3):
    if "metadata" not in message:
        message["metadata"] = {}
    return message["metadata"].get("error_count",0) >= max_error_count

def padding_messages_merge(data:List[Dict[str,Any]]):
    '''
    merge the neighbor messages with the same role
    '''
    temp_data = copy.deepcopy(data)
    padded_data = []
    last_role = None    
    for message in temp_data:
        if message["role"] == "system":
            padded_data.append(message)
            continue
        if last_role is None:
            if message["role"] == "assistant":
                padded_data.append({'content': 'continue', 'role': 'user'})                            
            padded_data.append(message)            
            last_role = message['role']
        elif last_role == message['role']:
            padded_data[-1]["content"] += f"\n{message['content']}"
        else:
            padded_data.append(message)            
            last_role = message['role']        
    if padded_data[-1]["role"] == "assistant":
        padded_data.append({'content': 'continue', 'role': 'user'})    
    return padded_data

def padding_messages_expand(data:Dict[str,Any]):
    '''
    padding the message between the neighbor messages with the same role
    '''
    temp_data = copy.deepcopy(data)
    padded_data = []        
    last_role = None                
    for message in temp_data:   
        if message["role"] == "system":
            padded_data.append(message)
            continue         
        if (last_role is None) and (message['role'] == 'assistant'):
            padded_data.append({'content': 'continue', 'role': 'user'})
            padded_data.append(message)

        elif (last_role is None) and (message['role'] == 'user'):                
            padded_data.append(message)    

        elif (last_role == message['role']) and (message['role'] == 'assistant'):
            padded_data.append({'content': 'continue', 'role': 'user'})
            padded_data.append(message)

        elif (last_role == message['role']) and (message['role'] == 'user'):
            padded_data.append({'content': 'continue', 'role': 'assistant'})
            padded_data.append(message)

        elif (last_role == message['role']) and (message['role'] == 'user'):                                        
            padded_data.append(message)

        else:
            padded_data.append(message)    
        
        last_role = message['role']
    
    if last_role == 'assistant':
        padded_data.append({'content': 'continue', 'role': 'user'})

    return padded_data