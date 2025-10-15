from typing import List, Dict, Any, Mapping
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



def merge_dict(*dicts: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple dictionaries with None-value filtering.
    
    This function merges multiple dictionaries into a single dictionary. The values
    of later dictionaries will only overwrite the values of earlier dictionaries if
    they are not None. This is useful for combining configuration dictionaries where
    None values should be ignored.
    
    Parameters
    ----------
    *dicts : Mapping[str, Any]
        Variable number of dictionaries or mapping objects to merge. Later
        dictionaries take precedence over earlier ones, but only for non-None values.
        
    Returns
    -------
    Dict[str, Any]
        A new dictionary containing the merged key-value pairs. Original
        dictionaries are not modified.
        
    Raises
    ------
    TypeError
        If any parameter is not a dictionary or mapping type.
        
    Examples
    --------
    >>> dict1 = {"a": 1, "b": 2, "c": None}
    >>> dict2 = {"b": 3, "c": 4, "d": None}
    >>> dict3 = {"c": 5, "e": 6}
    >>> merge_dict(dict1, dict2, dict3)
    {'a': 1, 'b': 3, 'c': 5, 'e': 6}
    
    >>> # None values are ignored
    >>> merge_dict({"a": 1}, {"a": None, "b": 2})
    {'a': 1, 'b': 2}
    
    >>> # Empty dictionaries are handled gracefully
    >>> merge_dict({}, {"a": 1}, {})
    {'a': 1}
    """
    result = {}
    for d in dicts:
        if not isinstance(d, Mapping):
            raise TypeError("All parameters must be dictionaries or mapping types")
        result.update({k: v for k, v in d.items() if v is not None})
    return result


def validate_required_params(params: Dict[str, Any], required_params: List[str]) -> None:
    """
    Validate that all required parameters are present in the params dictionary.
    
    This is a utility function for validating that all required parameters
    are present and not None in a parameters dictionary. Useful for API
    parameter validation across different LLM providers.
    
    Parameters
    ----------
    params : Dict[str, Any]
        The parameters dictionary to validate.
    required_params : List[str]
        List of required parameter names that must be present and not None.
        
    Raises
    ------
    ValueError
        If any required parameter is missing or None.
        
    Examples
    --------
    >>> params = {"messages": [...], "model": "gpt-4", "temperature": 0.7}
    >>> validate_required_params(params, ["messages", "model"])
    >>> # No error raised
    
    >>> params = {"messages": [...], "temperature": 0.7}
    >>> validate_required_params(params, ["messages", "model"])
    ValueError: Missing required parameters: model
    
    >>> params = {"messages": [...], "model": None, "temperature": 0.7}
    >>> validate_required_params(params, ["messages", "model"])
    ValueError: Missing required parameters: model
    """
    missing_params = [param for param in required_params if param not in params or params[param] is None]
    if missing_params:
        raise ValueError(f"Missing required parameters: {', '.join(missing_params)}")
