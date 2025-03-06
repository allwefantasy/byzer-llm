import functools
from byzerllm.apps.agent import get_agent_name
import inspect

def reply(agents=[]):
    def _impl(func):               
        @functools.wraps(func)
        def wrapper(self,*args, **kwargs): 
            _agents = agents                     
            is_lambda = inspect.isfunction(_agents) and agents.__name__ == '<lambda>'
            if is_lambda:
                _agents = agents(self)
            if len(_agents)==0 or get_agent_name(kwargs['sender']) in [get_agent_name(agent) for agent in _agents]:                                
                return func(*args, **kwargs)                
            return False, None
        wrapper._is_reply = True
        return wrapper
    return _impl

# def auto_register_reply(cls):
#     """
#     Class decorator to automatically register methods decorated by `reply`.
#     """
#     cls._reply_registry = {}  
#     for name, method in cls.__dict__.items():
#         if callable(method) and hasattr(method, '_is_reply'):
#             cls._reply_registry[name] = method
#     return cls


