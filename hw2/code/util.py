import time
from functools import wraps

def profile(func):
    @wraps(func)
    def with_profiling(*args, **kwargs):
        start_time = time.time()
        ret = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        print(f"[***]{func.__name__} executed {elapsed_time:.4f} sec")
        return ret
    return with_profiling
