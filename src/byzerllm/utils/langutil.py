import anyio
from concurrent.futures import ThreadPoolExecutor, TimeoutError, CancelledError
from threading import Event
from inspect import signature
from contextlib import contextmanager, closing
from contextlib2 import nullcontext
from functools import wraps, partial
import inspect
import os
import socket
from typing import Any, Optional
import re
from urllib.parse import urlparse
import warnings
import threading
import logging
import time

import json
from typing import Any, Dict, List, Union

@contextmanager
def switch_cwd(path):
    old_cwd = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(old_cwd)

@contextmanager
def patch(obj, attr, val):
    old_val = getattr(obj, attr)
    try:
        setattr(obj, attr, val)
        yield
    finally:
        setattr(obj, attr, old_val)


def asyncfy(func):
    """Decorator that makes a function async. Note that this does not actually make
    the function asynchroniously running in a separate thread, it just wraps it in
    an async function. If you want to actually run the function in a separate thread,
    consider using asyncfy_with_semaphore.

    Args:
        func (function): Function to make async
    """

    if inspect.iscoroutinefunction(func):
        return func

    @wraps(func)
    async def async_func(*args, **kwargs):
        return func(*args, **kwargs)

    return async_func


def asyncfy_with_semaphore(
    func, semaphore: Optional[anyio.Semaphore]=None, timeout: Optional[float] = None
):
    """Decorator that makes a function async, as well as running in a separate thread,
    with the concurrency controlled by the semaphore. If Semaphore is None, we do not
    enforce an upper bound on the number of concurrent calls (but it is still bound by
    the number of threads that anyio defines as an upper bound).

    Args:
        func (function): Function to make async. If the function is already async,
            this function will add semaphore and timeout control to it.
        semaphore (anyio.Semaphore or None): Semaphore to use for concurrency control.
            Concurrent calls to this function will be bounded by the semaphore.
        timeout (float or None): Timeout in seconds. If the function does not return
            within the timeout, a TimeoutError will be raised. If None, no timeout
            will be enforced. If the function is async, one can catch the CancelledError
            inside the function to handle the timeout.
    """
    if inspect.iscoroutinefunction(func):

        @wraps(func)
        async def async_func(*args, **kwargs):
            semaphore_ctx = semaphore if semaphore is not None else nullcontext()
            timeout_ctx = anyio.fail_after(timeout) if timeout else nullcontext()
            with timeout_ctx:
                async with semaphore_ctx:
                    return await func(*args, **kwargs)

        return async_func

    else:

        @wraps(func)
        async def async_func(*args, **kwargs):
            semaphore_ctx = semaphore if semaphore is not None else nullcontext()
            timeout_ctx = anyio.fail_after(timeout) if timeout else nullcontext()
            with timeout_ctx:
                async with semaphore_ctx:
                    return await anyio.to_thread.run_sync(
                        partial(func, *args, **kwargs), cancellable=True
                    )

        return async_func


def is_valid_url(candidate_str: Any) -> bool:
    if not isinstance(candidate_str, str):
        return False
    parsed = urlparse(candidate_str)
    return parsed.scheme != "" and parsed.netloc != ""


# backward compatible function name
def _is_valid_url(candidate_str: Any) -> bool:
    warnings.warn("_is_valid_url is deprecated. Please use is_valid_url instead.")
    return is_valid_url(candidate_str)


def _is_local_url(candidate_str: str) -> bool:
    parsed = urlparse(candidate_str)
    local_hosts = ["localhost", "127.0.0.1", "0.0.0.0", "::1"]
    return parsed.hostname in local_hosts


def find_available_port(port=None):
    if port is None:
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(("", 0))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return s.getsockname()[1]

    def is_port_occupied(port):
        """
        Returns True if the port is occupied, False otherwise.
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("localhost", port)) == 0

    while is_port_occupied(port):
        print(
            f"Port [yellow]{port}[/] already in use. Incrementing port number to",
            " find an available one."
        )
        port += 1
    return port

def to_bool(s: str) -> bool:
    """
    Convert a string to a boolean value.
    """
    if not isinstance(s, str):
        raise TypeError(f"Expected a string, got {type(s)}")
    true_values = ("yes", "true", "t", "1", "y", "on", "aye", "yea")
    false_values = ("no", "false", "f", "0", "n", "off", "nay", "")
    s = s.lower()
    if s in true_values:
        return True
    elif s in false_values:
        return False
    else:
        raise ValueError(
            f"Invalid boolean value: {s}. Valid true values: {true_values}. Valid false"
            f" values: {false_values}."
        )    

class CancellationRequested(Exception):
    """Raised when a task is requested to be cancelled."""
    pass


def run_in_thread(timeout: Optional[float] = None):
    """Decorator that runs a function in a thread with signal handling.
    
    Args:
        timeout (float, optional): Maximum time to wait for thread completion in seconds.
            If None, will wait indefinitely.
            
    The decorated function will run in a separate thread and can be interrupted by
    signals like Ctrl+C (KeyboardInterrupt). When interrupted, it will log the event
    and clean up gracefully.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                start_time = time.time()
                
                while True:
                    try:
                        # 使用较短的超时时间进行轮询，确保能够响应中断信号
                        poll_timeout = 0.1
                        if timeout is not None:
                            remaining = timeout - (time.time() - start_time)
                            if remaining <= 0:
                                future.cancel()
                                raise TimeoutError(f"Timeout after {timeout}s in {func.__name__}")
                            poll_timeout = min(poll_timeout, remaining)
                            
                        try:
                            return future.result(timeout=poll_timeout)
                        except TimeoutError:
                            continue  # 继续轮询
                            
                    except KeyboardInterrupt:
                        logging.warning("KeyboardInterrupt received, attempting to cancel task...")
                        future.cancel()
                        raise
                    except Exception as e:
                        logging.error(f"Error occurred in thread: {str(e)}")
                        raise
        return wrapper
    return decorator

def run_in_thread_with_cancel(timeout: Optional[float] = None):
    """Decorator that runs a function in a thread with explicit cancellation support.
    
    Args:
        timeout (float, optional): Maximum time to wait for thread completion in seconds.
            If None, will wait indefinitely.
            
    The decorated function MUST accept 'cancel_event' as its first parameter.
    This cancel_event is a threading.Event object that can be used to check if
    cancellation has been requested.
    
    The decorated function can be called with an external cancel_event passed as a keyword argument.
    If not provided, a new Event will be created.
    
    Example:
        @run_in_thread_with_cancel(timeout=10)
        def long_task(cancel_event, arg1, arg2):
            while not cancel_event.is_set():
                # do work
                if cancel_event.is_set():
                    raise CancellationRequested()
                    
        # 使用外部传入的cancel_event
        external_cancel = Event()
        try:
            result = long_task(arg1, arg2, cancel_event=external_cancel)
        except CancelledError:
            print("Task was cancelled")
            
        # 在其他地方取消任务
        external_cancel.set()
    """
    def decorator(func):
        # 检查函数签名
        sig = signature(func)
        params = list(sig.parameters.keys())
        if not params or params[0] != 'cancel_event':
            raise ValueError(
                f"Function {func.__name__} must have 'cancel_event' as its first parameter. "
                f"Current parameters: {params}"
            )
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 从kwargs中提取或创建cancel_event
            cancel_event = kwargs.pop('cancel_event', None) or Event()
            
            def cancellable_task():
                try:
                    return func(cancel_event, *args, **kwargs)
                except CancellationRequested:
                    logging.info(f"Task {func.__name__} was cancelled")
                    raise
                except Exception as e:
                    logging.error(f"Error in {func.__name__}: {str(e)}")
                    raise
            
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(cancellable_task)
                start_time = time.time()
                
                while True:
                    try:
                        # 使用较短的超时时间进行轮询，确保能够响应中断信号
                        poll_timeout = 0.1
                        if timeout is not None:
                            remaining = timeout - (time.time() - start_time)
                            if remaining <= 0:
                                cancel_event.set()
                                future.cancel()
                                raise TimeoutError(f"Timeout after {timeout}s in {func.__name__}")
                            poll_timeout = min(poll_timeout, remaining)
                            
                        try:
                            return future.result(timeout=poll_timeout)
                        except TimeoutError:
                            continue  # 继续轮询
                            
                    except KeyboardInterrupt:
                        logging.warning(f"KeyboardInterrupt received, cancelling {func.__name__}...")
                        cancel_event.set()
                        future.cancel()
                        raise CancelledError("Task cancelled by user")
                    except CancellationRequested:
                        logging.info(f"Task {func.__name__} was cancelled")
                        raise CancelledError("Task cancelled by request")
                    except Exception as e:
                        logging.error(f"Error occurred in thread: {str(e)}")
                        raise
                        
        return wrapper
    return decorator


def run_in_raw_thread():
    """A decorator that runs a function in a separate thread and handles exceptions.
    
    Args:
        func: The function to run in a thread
        
    Returns:
        A wrapper function that executes the decorated function in a thread
        
    The decorator will:
    1. Run the function in a separate thread
    2. Handle KeyboardInterrupt properly
    3. Propagate exceptions from the thread
    4. Support function arguments
    5. Preserve function metadata
    """
    def decorator(func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Store thread results
            result = []
            exception = []
            
            def worker():            
                ret = func(*args, **kwargs)
                result.append(ret)
            
            # Create and start thread with a meaningful name
            thread = threading.Thread(target=worker, name=f"{func.__name__}_thread")
            thread.daemon = True  # Make thread daemon so it doesn't prevent program exit
            
            try:
                thread.start()
                while thread.is_alive():
                    thread.join(0.1)

                return result[0] if result else None            
            except KeyboardInterrupt:                                
                raise KeyboardInterrupt("Task was cancelled by user")
                
        return wrapper
    return decorator    
