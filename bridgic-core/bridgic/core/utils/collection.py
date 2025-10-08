from typing import List, Dict, Any
from collections.abc import Hashable

def unique_list_in_order(ele_list: List[Any]) -> List[Any]:
    """
    Keep the order of the elements and remove the duplicates.
    """
    unique_ele = []
    seen = set()
    for ele in ele_list:
        if ele not in seen:
            seen.add(ele)
            unique_ele.append(ele)
    return unique_ele

def deep_hash(obj) -> int:
    """
    Recursively convert an object to a hashable form and calculate the hash value.
    """
    if isinstance(obj, (str, int, float, bool, type(None))):
        return hash(obj)
    elif isinstance(obj, (tuple, list)):
        return hash(tuple(deep_hash(e) for e in obj))
    elif isinstance(obj, dict):
        return hash(tuple(sorted((deep_hash(k), deep_hash(v)) for k, v in obj.items())))
    elif isinstance(obj, set):
        return hash(tuple(sorted(deep_hash(e) for e in obj)))
    elif isinstance(obj, Hashable):
        return hash(obj)
    else:
        raise TypeError(f"Unhashable type: {type(obj)}")