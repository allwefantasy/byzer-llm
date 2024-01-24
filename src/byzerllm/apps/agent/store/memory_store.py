from . import Message, MessageStore

class MemoryStore(MessageStore):
    def __init__(self):
        self.messages = {}

    def put(self, message: Message):
        if message.id not in self.messages:
            self.messages[message.id] = []
        self.messages[message.id].append(message)
        return self
    
    def get(self, id: str):
        return self.messages.get(id,[])        
    