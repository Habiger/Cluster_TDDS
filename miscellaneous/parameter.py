import json
from dataclasses import dataclass, is_dataclass, asdict

def nested_dataclass(*args, **kwargs):
    """Enables initialization of inner dataclasses through a nested dictionary.
    """
    def wrapper(cls):
        cls = dataclass(cls, **kwargs)
        original_init = cls.__init__
        def __init__(self, *args, **kwargs):
            for name, value in kwargs.items():
                field_type = cls.__annotations__.get(name, None)
                if is_dataclass(field_type) and isinstance(value, dict):
                     new_obj = field_type(**value)
                     kwargs[name] = new_obj
            original_init(self, *args, **kwargs)
        cls.__init__ = __init__
        return cls
    return wrapper(args[0]) if args else wrapper

@dataclass
class Parameter():
    """baseclass for parameter dataclasses
    """
    def __init__(self): pass
    
    def get_dict(self):
        """`Note:` Also nested dataclasses will be returned as a dict.
        """
        return asdict(self)

    def __str__(self):
        return json.dumps(self.get_dict(), indent=4)
