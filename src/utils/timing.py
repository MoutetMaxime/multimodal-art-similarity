import time
from functools import wraps


class TimingConfig:
    ENABLE = True

def timing(method):
    @wraps(method)
    def timed(*args, **kwargs):
        if TimingConfig.ENABLE:
            start = time.time()
            result = method(*args, **kwargs)
            end = time.time()
            print(f"[TIMER] {method.__qualname__} took {end - start:.4f} seconds")
            return result
        else:
            return method(*args, **kwargs)
    return timed
