# Langutil Utility Functions Documentation

This document provides an overview and usage examples of the utility functions found in `src/byzerllm/utils/langutil.py`. These functions primarily help with asynchronous execution, URL validation, thread management, and more. Below are concise explanations along with multiple usage examples for different cases.

---

## Table of Contents

1. [Context Managers](#context-managers)
2. [Asynchronous Helpers](#asynchronous-helpers)
3. [URL Validation](#url-validation)
4. [Conversion Utilities](#conversion-utilities)
5. [Thread Management](#thread-management)
6. [Exception Handling](#exception-handling)

---

## Context Managers

### `switch_cwd`

This context manager temporarily switches the current working directory to the given path and then reverts it.

**Usage Example:**

```python
import os
from byzerllm.utils.langutil import switch_cwd

print('Current directory:', os.getcwd())
with switch_cwd('/tmp'):
    print('Inside context directory:', os.getcwd())
print('Reverted directory:', os.getcwd())
```

### `patch`

Temporarily patch an attribute of an object within the context.

**Usage Example:**

```python
from byzerllm.utils.langutil import patch

class MyClass:
    value = 10

obj = MyClass()
print('Original value:', obj.value)
with patch(MyClass, 'value', 20):
    print('Patched value:', obj.value)
print('Reverted value:', obj.value)
```

---

## Asynchronous Helpers

### `asyncfy`

A decorator that wraps a synchronous function into an asynchronous one. This does not run the function in another thread but allows you to await it.

**Usage Example:**

```python
import asyncio
from byzerllm.utils.langutil import asyncfy

@asyncfy
def greet(name):
    return f"Hello, {name}!"

async def main():
    result = await greet('Alice')
    print(result)

asyncio.run(main())
```

### `asyncfy_with_semaphore`

Wraps a function to run asynchronously using thread execution while enforcing concurrency control with an optional semaphore and timeout.

**Usage Example with a Semaphore:**

```python
import asyncio
import anyio
from byzerllm.utils.langutil import asyncfy_with_semaphore

# Create a semaphore to limit concurrent execution
semaphore = anyio.Semaphore(2)

@asyncfy_with_semaphore
def compute(x, y):
    return x + y

async def main():
    results = await asyncio.gather(
        compute(1, 2, semaphore=semaphore),
        compute(3, 4, semaphore=semaphore)
    )
    print('Results:', results)

asyncio.run(main())
```

---

## URL Validation

### `is_valid_url`

Checks whether a given string is a valid URL.

**Usage Example:**

```python
from byzerllm.utils.langutil import is_valid_url

url = "https://www.example.com"
if is_valid_url(url):
    print(f'URL {url} is valid.')
else:
    print(f'URL {url} is not valid.')
```

### `_is_valid_url` (Deprecated)

A backward-compatible function with a deprecation warning. Use `is_valid_url` instead.

---

## Conversion Utilities

### `to_bool`

Converts a string to a boolean value. Accepts various representations of true/false.

**Usage Example:**

```python
from byzerllm.utils.langutil import to_bool

print(to_bool('yes'))  # Outputs: True
print(to_bool('no'))   # Outputs: False
```

---

## Thread Management

### `run_in_thread`

Decorator that runs a function in a separate thread with signal handling. It uses a polling mechanism to support timeouts and respond to interrupts gracefully.

**Usage Example:**

```python
from byzerllm.utils.langutil import run_in_thread
import time

@run_in_thread(timeout=5)
def long_running_task(x):
    for i in range(10):
        print(f'Processing {i} with parameter {x}')
        time.sleep(1)
    return 'Task completed.'

# Call the task
result = long_running_task(100)
print(result)
```

### `run_in_thread_with_cancel`

Similar to `run_in_thread`, but supports explicit cancellation via a `cancel_event`. The decorated function must accept a `cancel_event` argument.

**Usage Example:**

```python
import time
from threading import Event
from byzerllm.utils.langutil import run_in_thread_with_cancel, CancellationRequested

@run_in_thread_with_cancel(timeout=10)
def cancellable_task(cancel_event, count):
    for i in range(count):
        if cancel_event.is_set():
            raise CancellationRequested()
        print(f'Working on step {i}')
        time.sleep(1)
    return 'Cancellable task finished.'

external_cancel = Event()
try:
    # In a real scenario, you might cancel this from another thread
    result = cancellable_task(5, cancel_event=external_cancel)
    print(result)
except Exception as e:
    print('Task was cancelled or timed out:', str(e))
```

### `run_in_raw_thread`

Decorator that runs a function in a separate thread and handles exceptions, including proper propagation of exceptions.

**Usage Example:**

```python
from byzerllm.utils.langutil import run_in_raw_thread

@run_in_raw_thread()
def raw_task(msg):
    print(f'Running task with message: {msg}')
    return 'Raw task completed.'

result = raw_task('Hello from raw thread')
print(result)
```

---

## Exception Handling

### `CancellationRequested`

An exception that is raised when a task is requested to be canceled, particularly useful in contexts that support cancellation through threading.

**Usage Example:**

```python
from byzerllm.utils.langutil import CancellationRequested

try:
    raise CancellationRequested()
except CancellationRequested:
    print('Caught the cancellation exception successfully.')
```

---

## Summary

The utilities provided in `langutil.py` assist in managing asynchronous executions, running tasks in threads with various safeguards, temporarily modifying execution contexts, and performing common validations and conversions. Use the examples above as a starting point to integrate these tools into your applications effectively.
