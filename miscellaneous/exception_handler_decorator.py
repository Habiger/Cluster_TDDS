import functools


def catch_exceptions(logger):
    """
    A decorator that wraps the passed in function and logs 
    exceptions should one occur
    
    @param logger: The logging object
    """
    
    def decorator(func):
    
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except:
                # log the exception
                err = "There was an exception in  "
                err += func.__name__
                logger.exception(err)
            
            # returns None
            return None
        return wrapper
    return decorator