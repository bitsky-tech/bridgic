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

def filter_dict(data: Dict[str, Any], exclude_none: bool = True, exclude_values: tuple = ()) -> Dict[str, Any]:
    """
    Filter a dictionary by removing keys with specific values.
    
    Parameters
    ----------
    data : Dict[str, Any]
        The dictionary to filter.
    exclude_none : bool, optional
        If True, remove keys with None values (default is True).
    exclude_values : tuple, optional
        Additional values to exclude. Keys with these values will be removed.
        
    Returns
    -------
    Dict[str, Any]
        A new dictionary with filtered key-value pairs.
        
    Examples
    --------
    >>> filter_dict({"a": 1, "b": None, "c": 3})
    {"a": 1, "c": 3}
    
    >>> from openai import omit
    >>> filter_dict({"a": 1, "b": omit, "c": None}, exclude_values=(omit,))
    {"a": 1}
    """
    filtered = {}
    for key, value in data.items():
        # Skip None values if exclude_none is True
        if exclude_none and value is None:
            continue
        # Skip values in exclude_values tuple
        if exclude_values and any(value is excluded_val for excluded_val in exclude_values):
            continue
        filtered[key] = value
    return filtered
