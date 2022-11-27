import functools
import logging

def catch_exceptions(logger: logging.Logger):
    """A decorator that catches and logs exceptions should one occur.
    """
    def decorator(func):
    
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except:
                # log the exception
                err = "\nThere was an exception in  "
                err += func.__name__
                
                logger.exception(err)
            return None
        return wrapper
    return decorator