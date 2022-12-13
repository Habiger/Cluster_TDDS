from functools import wraps
import logging

def catch_exceptions(logger: logging.Logger):
    """A decorator that catches and logs exceptions should one occur.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:                
                logger.exception(f"\n\nException raised in {func.__name__}.\n Exception: {str(e)}")
            return None
        return wrapper
    return decorator