import collections.abc

""""currently not used; to delete?!                #TODO
"""


def update(og_dict, update_dict):
    """Recursively updates the entries in a possibly nested dict. \\
        `Note:`Does not raise an error if there are new/wrong keys updated.
    """
    for k, v in update_dict.items():
        if isinstance(v, collections.abc.Mapping):
            og_dict[k] = update(og_dict.get(k, {}), v)
        else:
            og_dict[k] = v
    return og_dict
