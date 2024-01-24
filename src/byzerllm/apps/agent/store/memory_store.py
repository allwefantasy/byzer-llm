from . import Message, MessageStore
import time

class MemoryStore(MessageStore):
    def __init__(self):
        self.messages = {}

    def put(self, message: Message):
        if message.id not in self.messages:
            self.messages[message.id] = []
        v = self.messages[message.id]
        v.append(message)
       
        # clean up old messages
        target = -1 
        for idx,item in enumerate(v):
            if time.monotonic() - item.timestamp > 24*60*60:
                target = idx
                break
        if target >= 0:
            v = v[target:]
            self.messages[message.id] = v
        
        # clean up message groups
        remove_keys = []
        for key in list(self.messages.keys()):
            if time.monotonic() -  self.messages[key][-1].timestamp  > 24*60*60:
                remove_keys.append(key)

        for key in remove_keys:
            del self.messages[key]        

        return self
    
    def get(self, id: str):
        return self.messages.get(id,[])        
    