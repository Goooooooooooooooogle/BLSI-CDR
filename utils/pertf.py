import time

from functools import wraps

def func_exec_time(func):
    @wraps(func)  
    def wrapper(*args,**kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        return result
    return wrapper
