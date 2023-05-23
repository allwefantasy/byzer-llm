from timeout_decorator import timeout
import time

def run_function_with_timeout(timeout_duration):
    def decorator(func):
        @timeout(timeout_duration)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Define your function to be executed
def my_function():
    time.sleep(1)

# Set the desired timeout duration dynamically
timeout_duration = 10  # Timeout duration in seconds

# Apply the decorator with the dynamically set timeout duration
decorated_function = run_function_with_timeout(timeout_duration)(my_function)

try:
    decorated_function()
except TimeoutError:
    # Code to handle the timeout
    print("Function execution timed out.")