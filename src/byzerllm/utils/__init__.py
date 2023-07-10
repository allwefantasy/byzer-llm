from pathlib import Path
from typing import Any, List, Optional,Union

def print_flush(*args, **kwargs):
    print(*args, **kwargs, flush=True)

import signal
from contextlib import contextmanager
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException()

@contextmanager
def timeout(duration: float):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    try:
        yield
    finally:
        signal.alarm(0)