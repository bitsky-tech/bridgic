from typing import List, Dict, Any

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