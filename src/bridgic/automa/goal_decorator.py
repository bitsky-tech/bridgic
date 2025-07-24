from typing import List, Callable, Union
from typing_extensions import overload
from bridgic.types.common_types import PromptTemplate

@overload
def goal(
    *,
    preconditions: List[str] = [],
    final: bool = False,
    priority: int = 0,
) -> Callable:
    """
    A decorator that marks a method as a goal within an GaopAutoma class. 

    An Automa can have multiple goals. The framework will process goals in descending order of priority, planning and executing paths for each goal. Once a goal marked as final is achieved, the Automa stops running.
    
    Parameters
    ----------
    preconditions : List[str]
        The preconditions required for achieving the current goal, expressed as a list of precondition IDs. The framework will automatically extract preconditions from the decorated method, and the preconditions specified here will be merged with the extracted ones.
    final : bool
        Indicates if this goal is a final goal. When set to True, it has two effects: 1) The Automa will stop executing immediately once this goal is achieved, and 2) The return value from the decorated method will be used as the Automa's output (via process_async()).
    priority : int
        Specifies the priority of this goal. Goals with higher priority numbers will be planned and achieved first.
   """
    ...

@overload
def goal(
    *,
    description: PromptTemplate
) -> type:
    """
    A decorator that marks a LlmpAutoma class, specifying the Automa's goal. 

    Each LlmpAutoma can only have one goal specified. Multiple goals are not supported.
    
    Parameters
    ----------
    description : PromptTemplate
        Specifies the goal of LlmpAutoma in natural language form. This goal can be described using a string or a more expressive prompt template. This parameter is required.
   """
    ...

def goal_goap(
    func: Callable,
    *,
    preconditions: List[str] = [],
    final: bool = False,
    priority: int = 0,
) -> Callable:
    func.__goal_config__ = {
        "preconditions": preconditions,
        "final": final,
        "priority": priority,
    }
    return func

def goal_llmp(
    cls: type,
    *,
    description: PromptTemplate
) -> type:
    cls.__goal_config__ = {
        "description": description,
    }
    return cls

def goal(**kwargs) -> Union[Callable, type]:
    """
    The implementation of the overloaded goal decorators defined above.
    """
    def wrapper(cls_or_func):
        if isinstance(cls_or_func, type):
            # Expect: goal is decorating the LlmpAutoma class
            return goal_llmp(
                cls_or_func,
                **kwargs
            )
        else:
            # Expect: goal is decorating a GoapAutoma method
            return goal_goap(
                cls_or_func,
                **kwargs
            )
    return wrapper
