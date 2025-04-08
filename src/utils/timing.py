import time
from functools import wraps

ENABLE_TIMING = True

def timing(method):
    @wraps(method)
    def timed(*args, **kw):
        if ENABLE_TIMING:
            start = time.time()
            result = method(*args, **kw)
            end = time.time()
            print(f"[TIMER] {method.__qualname__} took {end - start:.4f} seconds")
            return result
        else:
            return method(*args, **kw)
    return timed
