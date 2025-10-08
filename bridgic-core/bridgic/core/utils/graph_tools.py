from typing import Dict, List
from collections import defaultdict, deque
from bridgic.core.types.error import *

def validate_dag_constraints(forward_dict: Dict[str, List[str]]):
    """
    Use Kahn's algorithm to check if the input graph described by the forward_dict satisfies
    the DAG constraints. If the graph doesn't meet the DAG constraints, AutomaDeclarationError 
    will be raised. 

    More about [Kahn's algorithm](https://en.wikipedia.org/wiki/Topological_sorting#Kahn's_algorithm)
    could be read from the link.

    Parameters
    ----------
    forward_dict : Dict[str, List[str]]
        A dictionary that describes the graph structure. The keys are the nodes, and the values 
        are the lists of nodes that are directly reachable from the keys.

    Raises
    ------
    AutomaDeclarationError
        If the graph doesn't meet the DAG constraints.
    """
    # 1. Initialize the in-degree.
    in_degree = defaultdict(int)
    for current, target_list in forward_dict.items():
        for target in target_list:
            in_degree[target] += 1

    # 2. Create a queue of workers with in-degree 0.
    queue = deque([node for node in forward_dict.keys() if in_degree[node] == 0])

    # 3. Continuously pop workers from the queue and update the in-degree of their targets.
    while queue:
        node = queue.popleft()
        for target in forward_dict.get(node, []):
            in_degree[target] -= 1
            if in_degree[target] == 0:
                queue.append(target)

    # 4. If the in-degree were all 0, then the graph meets the DAG constraints.
    if not all([in_degree[node] == 0 for node in in_degree.keys()]):
        nodes_in_cycle = [node for node in forward_dict.keys() if in_degree[node] != 0]
        raise AutomaCompilationError(
            f"the graph automa does not meet the DAG constraints, because the "
            f"following workers are in cycle: {nodes_in_cycle}"
        )