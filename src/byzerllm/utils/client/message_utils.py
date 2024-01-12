from typing import List, Dict,Any
import copy

def padding_messages_merge(data:List[Dict[str,Any]]):
    '''
    merge the neighbor messages with the same role
    '''
    temp_data = copy.deepcopy(data)
    padded_data = []
    last_role = None    
    for message in temp_data:
        if last_role is None:
            padded_data.append(message)            
            last_role = message['role']
        elif last_role == message['role']:
            padded_data[-1]["content"] += f"\n{message['content']}"
        else:
            padded_data.append(message)            
            last_role = message['role']        
        
    return padded_data

def padding_messages_expand(data:Dict[str,Any]):
    '''
    padding the message between the neighbor messages with the same role
    '''
    temp_data = copy.deepcopy(data)
    padded_data = []        
    last_role = None                
    for message in temp_data:            
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