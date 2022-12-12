import json
import os

def save_dict_as_json(path, data, filename):
    with open(os.path.join(path, filename), 'w') as fp:
        json.dump(data, fp, indent=4)
