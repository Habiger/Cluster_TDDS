import json
import os

from dataclasses import dataclass, is_dataclass, asdict

def nested_dataclass(*args, **kwargs):
    """Decorator which enables initialization of inner dataclasses through a nested dictionary.
    """
    def wrapper(cls):
        cls = dataclass(cls, **kwargs)
        original_init = cls.__init__
        
        def __init__(self, *args, **kwargs):
            post_init = None
            for name in list(kwargs.keys()): # had to make a list out of keys, else there is a runtime error because dict changes size during iteration (-> kwarg.pop())
                value = kwargs[name]  
                field_type = cls.__annotations__.get(name, None)
                if is_dataclass(field_type) and isinstance(value, dict):
                    if cls.__dataclass_fields__[name].init:  # checks if field has init == True
                        new_obj = field_type(**value)
                        kwargs[name] = new_obj
                    else:                                    # stores kwargs for post_init
                        post_init = cls.__post_init__
                        kwargs_post_init = kwargs.pop(name)
            original_init(self, *args, **kwargs)
            if post_init:
                post_init(self, **kwargs_post_init)
        cls.__init__ = __init__
        return cls
    return wrapper(args[0]) if args else wrapper


@dataclass
class Parameter:
    """baseclass for parameter dataclass
    """
    def __init__(self):
        pass
    
    def get_dict(self):
        """`Note:` Also nested dataclasses will be returned as a dict.
        """
        return asdict(self)

    def __str__(self):
        return json.dumps(self.get_dict(), indent=4)

    def save(self, path):
        """saves Parameter object as a json file at given path\\
            the name will be the name of the parameter class

        Args:
            path (str): path where to save file
        """
        filepath = os.path.join(path, f"{type(self).__name__}.json")
        with open(filepath, 'w') as fp:
            json.dump(self.get_dict(), fp, indent=4)
