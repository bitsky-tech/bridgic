import enum
import inspect
from inspect import Parameter
from copy import deepcopy
from dataclasses import dataclass
from typing import Tuple, Any, List, Dict, Mapping, Union, Callable
from types import MethodType
from typing_extensions import TYPE_CHECKING

from bridgic.core.types._common import ArgsMappingRule, ResultDispatchingRule
from bridgic.core.types._error import WorkerArgsMappingError
from bridgic.core.automa.args._args_descriptor import WorkerInjector

if TYPE_CHECKING:
    from bridgic.core.automa._graph_automa import (
        GraphAutoma,
        _GraphAdaptedWorker,
        _AddWorkerDeferredTask,
        _RemoveWorkerDeferredTask,
        _AddDependencyDeferredTask,
    )

@dataclass
class InOrder:
    """
    A descriptor to indicate that data should be distributed to multiple workers. 

    When is used to input arguments or worker with this descriptor, the data will be distributed
    to downstream workers instead of being gathered as a single value. Split the returned 
    Sequence object and dispatching them in-order and element-wise to the downstream workers 
    as their actual input.

    Parameters
    ----------
    data : Union[List, Tuple]
        The data to be distributed. Must be a list or tuple with length matching
        the number of workers that will receive it.

    Raises
    ------
    ValueError
        If the data is not a list or tuple.
    """
    data: Union[List[Any], Tuple[Any], inspect._empty] = None

    def check_data(self) -> None:
        if not isinstance(self.data, (List, Tuple)):
            raise ValueError(f"The data of the Distribute must be `List` or `Tuple`, but got Type `{type(self.data)}`.")


class ArgsManager:
    """
    Manages argument binding, inputs propagation, and arguments injection between 
    workers in a graph automa.
    """
    def __init__(
        self,
        input_args: Tuple[Any, ...],
        input_kwargs: Dict[str, Any],
        worker_outputs: Dict[str, Any],
        worker_forwards: Dict[str, List[str]],
        worker_dict: Dict[str, "_GraphAdaptedWorker"],
    ):
        """
        Initialize the ArgsManager with worker graph information.

        Parameters
        ----------
        input_args : Tuple[Any, ...]
            Initial positional arguments passed to the automa.
        input_kwargs : Dict[str, Any]
            Initial keyword arguments passed to the automa.
        worker_outputs : Dict[str, Any]
            Dictionary mapping worker keys to their output values.
        worker_forwards : Dict[str, List[str]]
            Dictionary mapping each worker key to the list of workers it triggers.
        worker_dict : Dict[str, "_GraphAdaptedWorker"]
            Dictionary mapping worker keys to their worker objects with binding information.
        """
        # record the args that are possibly passed to the workers
        self._input_args = input_args
        self._input_kwargs = input_kwargs
        self._worker_outputs = worker_outputs
        self._start_arguments = {
            **{f"__arg_{i}": arg for i, arg in enumerate(input_args)},
            **{f"__kwarg_{k}": v for k, v in input_kwargs.items()}
        }

        # record the forward count and index of the worker outputs
        start_worker_keys = [key for key, worker in worker_dict.items() if worker.is_start]
        self._worker_forward_count = {
            **{key: {
                "forward_count": len(value),  # how many workers are triggered by the current worker
                "forward_index": 0,  # the index of the current output to distribute
            } for key, value in worker_forwards.items()},
            **{key: {
                "forward_count": len(start_worker_keys),
                "forward_index": 0,
            } for key, _ in self._start_arguments.items()}
        }

        # record the receiver and sender rules of the workers
        self._worker_rule_dict = {}
        for key, worker in worker_dict.items():
            receiver_rule, sender_rule = worker.args_mapping_rule, worker.result_dispatching_rule
            dependencies = deepcopy(worker.dependencies)
            if worker.is_start:
                dependencies.append("__automa__")

            self._worker_rule_dict[key] = {
                "dependencies": dependencies,
                "param_names": worker.get_input_param_names(),
                "receiver_rule": receiver_rule,
                "sender_rule": sender_rule,
            }
        for key, value in self._start_arguments.items():
            if isinstance(value, InOrder):
                sender_rule = ResultDispatchingRule.IN_ORDER
            else:
                sender_rule = ResultDispatchingRule.AS_IS
            self._worker_rule_dict[key] = {
                "dependencies": [],
                "param_names": [],
                "receiver_rule": None,
                "sender_rule": sender_rule,
            }

        self._injector = WorkerInjector()

    ###############################################################################
    # Arguments Binding between workers that have dependency relationships.
    ###############################################################################
    def args_binding(
        self, 
        last_worker_key: str, 
        current_worker_key: str,
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        """
        Bind arguments from a predecessor worker to the current worker.

        This method handles argument binding between workers that have dependency
        relationships. It supports two scenarios:
        1. Start workers: binding initial input arguments to start workers
        2. Dependent workers: binding outputs from dependency workers to the current worker

        The binding process respects the receiver rule of the current worker and
        the sender rules of the predecessor workers.

        Parameters
        ----------
        last_worker_key : str
            The key of the predecessor worker. Use "__automa__" for start workers
            that should receive initial input arguments.
        current_worker_key : str
            The key of the current worker that will receive the arguments.

        Returns
        -------
        Tuple[Tuple[Any, ...], Dict[str, Any]]
            A tuple containing (positional_args, keyword_args) to be passed to
            the current worker. Returns empty arguments if last_worker_key is not
            a dependency of current_worker_key.
        """
        kickoff_single = "start" if last_worker_key == "__automa__" else "dependency"
        worker_dependencies = self._worker_rule_dict[current_worker_key]["dependencies"]
        worker_receiver_rule = self._worker_rule_dict[current_worker_key]["receiver_rule"]

        # If the last worker is not a dependency of the current worker, then return empty arguments.
        if not last_worker_key in worker_dependencies:
            return (), {}
    
        def _start_args_binding() -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
            data_mode_list = []
            for key, value in self._start_arguments.items():
                sender_rule = self._worker_rule_dict[key]["sender_rule"]
                if isinstance(value, InOrder) and value.data is None:
                    raise ValueError(f"The data of the Dispatching must be `List` or `Tuple`, but got Type `{type(value.data)}`.")

                data_mode_list.append({
                    'worker_key': key,
                    'data': value.data if isinstance(value, InOrder) else value,
                    'send_rule': sender_rule,
                })
            data = self._args_send(data_mode_list)
            next_args = tuple([
                data_item
                for data_mode, data_item in zip(data_mode_list, data)
                if data_mode['worker_key'].startswith('__arg')
            ])
            next_kwargs = {
                data_mode['worker_key'].strip('__kwarg_'): data_item
                for data_mode, data_item in zip(data_mode_list, data)
                if data_mode['worker_key'].startswith('__kwarg')
            }
            return next_args, next_kwargs

        def _dependency_args_binding(
            worker_dependencies: List[str], 
            worker_receiver_rule: ArgsMappingRule, 
        ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
            data_mode_list = []
            for dependency_worker_key in worker_dependencies:
                dependency_worker_output = self._worker_outputs[dependency_worker_key]
                dependency_worker_send_rule = self._worker_rule_dict[dependency_worker_key]["sender_rule"]
                data_mode_list.append({
                    'worker_key': dependency_worker_key,
                    'data': dependency_worker_output,
                    'send_rule': dependency_worker_send_rule,
                })
            data = self._args_send(data_mode_list)
            return self._args_receive(last_worker_key, current_worker_key, worker_receiver_rule, data)

        if kickoff_single == "start":
            next_args, next_kwargs = _start_args_binding()
        elif kickoff_single == "dependency":
            next_args, next_kwargs = _dependency_args_binding(worker_dependencies, worker_receiver_rule)

        # There is a situation where the next worker is automa, and the parameters it receives sometimes 
        # need to be distributed. In this case, this information is contained in the parameter declaration 
        # of automa rather than in the result of the previous worker. So we need to deal with this situation.
        params_name = self._worker_rule_dict[current_worker_key]["param_names"]
        params_filled_arguments = get_filled_params(next_args, next_kwargs, params_name)
        for _, params in params_name.items():
            for param in params:
                param_name, param_type = param
                if isinstance(param_type, InOrder):
                    param_info = params_filled_arguments[param_name]
                    if param_info['source'] == 'args':
                        next_args_list = list(next_args)
                        distribute_data = InOrder(data=param_info['value'])
                        distribute_data.check_data()
                        next_args_list[param_info['index']] = distribute_data
                        next_args = tuple(next_args_list)
                    elif param_info['source'] == 'kwargs':
                        distribute_data = InOrder(data=param_info['value'])
                        distribute_data.check_data()
                        next_kwargs[param_info['key']] = distribute_data
        
        return next_args, next_kwargs

    def _args_send(self, data_mode_list: List[Dict[str, Any]]) -> List[Any]:
        """
        Process and send data according to sender rules.

        This method applies sender rules (GATHER or DISTRIBUTE) to prepare data
        for downstream workers. With GATHER, the entire output is sent. With
        DISTRIBUTE, elements are distributed one at a time to different workers.

        Parameters
        ----------
        data_mode_list : List[Dict[str, Any]]
            List of dictionaries, each containing 'worker_key', 'data', and 'send_rule'.

        Returns
        -------
        List[Any]
            List of processed data items ready to be sent to downstream workers.

        Raises
        ------
        WorkerArgsMappingError
            If the data is not iterable when DISTRIBUTE rule is used.
            If the data length doesn't match the forward count for DISTRIBUTE.
            If an unsupported sender rule is encountered.
        """
        send_data = []
        for data_mode in data_mode_list:
            worker_key = data_mode['worker_key']
            data = data_mode['data']
            send_rule = data_mode['send_rule']
            if send_rule == ResultDispatchingRule.AS_IS:
                send_data.append(data)
            elif send_rule == ResultDispatchingRule.IN_ORDER:
                # if the worker output is not iterable -- tuple or list, then raise error
                if not isinstance(data, (tuple, list)):
                    raise WorkerArgsMappingError(
                        f"The worker's output of '{worker_key}' is not iterable to distribute, "
                        f"but got type {type(data)}."
                    )

                # if the worker output is less than the forward count, then raise error
                if len(data) != self._worker_forward_count[worker_key]["forward_count"]:
                    raise WorkerArgsMappingError(
                        f"The worker's output of '{worker_key}' has not the same output count as the worker that are triggered by it, "
                        f"there should be {self._worker_forward_count[worker_key]['forward_count']} output, "
                        f"but got {len(data)} output."
                    )

                # get the index of the output to distribute and increment the forward index
                idx = self._worker_forward_count[worker_key]["forward_index"]
                send_data.append(data[idx])
                self._worker_forward_count[worker_key]["forward_index"] += 1
            else:
                raise WorkerArgsMappingError(
                    f"The sender rule of the worker '{worker_key}' is not supported."
                )
        return send_data

    def _args_receive(
        self, 
        last_worker_key: str,
        current_worker_key: str,
        current_worker_receiver_rule: ArgsMappingRule,
        data: List[Any]
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        """
        Process received data according to the current worker's receiver rule.

        This method applies receiver rules to transform data from predecessor workers
        into the appropriate argument format for the current worker. Supported rules:
        - AS_IS: Pass all data as positional arguments
        - UNPACK: Unpack a single result (list/tuple/dict) into arguments
        - MERGE: Wrap all results into a single tuple argument
        - SUPPRESSED: Ignore all data and return empty arguments

        Parameters
        ----------
        last_worker_key : str
            The key of the predecessor worker that produced the data.
        current_worker_key : str
            The key of the current worker that will receive the arguments.
        current_worker_receiver_rule : ArgsMappingRule
            The receiver rule to apply when processing the data.
        data : List[Any]
            The data received from predecessor workers.
        
        Returns
        -------
        Tuple[Tuple[Any, ...], Dict[str, Any]]
            A tuple containing (positional_args, keyword_args) ready to be passed
            to the current worker.

        Raises
        ------
        WorkerArgsMappingError
            If UNPACK rule is used but the worker has multiple dependencies.
            If UNPACK rule is used but the data type is not unpackable.
        """
        def as_is_return_values(results: List[Any]) -> Tuple[Tuple, Dict[str, Any]]:
            next_args, next_kwargs = tuple(results), {}
            return next_args, next_kwargs

        def unpack_return_value(result: Any) -> Tuple[Tuple, Dict[str, Any]]:
            # result is not allowed to be None, since None can not be unpacked.
            if isinstance(result, (List, Tuple)):
                # Similar args mapping logic to as_is_return_values()
                next_args, next_kwargs = tuple(result), {}
            elif isinstance(result, Mapping):
                next_args, next_kwargs = (), {**result}

            else:
                # Other types, including None, are not unpackable.
                raise WorkerArgsMappingError(
                    f"args_mapping_rule={ArgsMappingRule.UNPACK} is only valid for "
                    f"tuple/list, or dict. But the worker '{current_worker_key}' got type '{type(result)}' from the last worker '{last_worker_key}'."
                )
            return next_args, next_kwargs

        def merge_return_values(results: List[Any]) -> Tuple[Tuple, Dict[str, Any]]:
            next_args, next_kwargs = tuple([results]), {}
            return next_args, next_kwargs

        if current_worker_receiver_rule == ArgsMappingRule.AS_IS:
            next_args, next_kwargs = as_is_return_values(data)
        elif current_worker_receiver_rule == ArgsMappingRule.UNPACK:
            if len(data) != 1:
                raise WorkerArgsMappingError(
                    f"The worker must has exactly one dependency for the args_mapping_rule=\"{ArgsMappingRule.UNPACK}\", "
                    f"but got dependencies: {last_worker_key}, and the data is {data}"
                )
            next_args, next_kwargs = unpack_return_value(*data)
        elif current_worker_receiver_rule == ArgsMappingRule.MERGE:
            next_args, next_kwargs = merge_return_values(data)
        elif current_worker_receiver_rule == ArgsMappingRule.SUPPRESSED:
            next_args, next_kwargs = (), {}

        return next_args, next_kwargs

    ###############################################################################
    # Arguments Binding of Inputs Arguments.
    ###############################################################################
    def inputs_propagation(
        self,
        current_worker_key: str,
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        """
        Propagate input arguments from the automa to a worker.

        This method filters and propagates keyword arguments from the automa's
        input buffer to a worker, but only for parameters that can accept keyword
        arguments (positional_or_keyword or positional_only parameters).

        Parameters
        ----------
        current_worker_key : str
            The key of the worker that should receive the propagated inputs.

        Returns
        -------
        Tuple[Tuple[Any, ...], Dict[str, Any]]
            A tuple containing empty positional arguments and a dictionary of
            keyword arguments that match the worker's parameter signature.
        """
        input_kwargs = {k:v for k,v in self._input_kwargs.items()}
        rx_param_names_dict = self._worker_rule_dict[current_worker_key]["param_names"]

        def get_param_names(param_names_dict: List[Tuple[str, Any]]) -> List[str]:
            return [name for name, _ in param_names_dict]
            
        positional_only_param_names = get_param_names(rx_param_names_dict.get(Parameter.POSITIONAL_ONLY, []))
        positional_or_keyword_param_names = get_param_names(rx_param_names_dict.get(Parameter.POSITIONAL_OR_KEYWORD, []))
        var_keyword_param_names = get_param_names(rx_param_names_dict.get(Parameter.VAR_KEYWORD, []))

        propagation_kwargs = {}
        for key, value in input_kwargs.items():
            if var_keyword_param_names:
                propagation_kwargs[key] = value
                continue

            if (
                key in positional_only_param_names or
                key in positional_or_keyword_param_names
            ):
                propagation_kwargs[key] = value

        return (), propagation_kwargs

    ###############################################################################
    # Arguments Injection for workers that need data from other workers that not directly depend on it.
    ###############################################################################
    def args_injection(
        self,
        current_worker_key: str,
        current_automa: "GraphAutoma",
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        """
        Inject arguments for workers that need data from non-dependency workers.

        This method handles special argument injection using descriptors like `From()`
        and `System()`, which allow workers to access data from workers they don't
        directly depend on, or to access system-level resources.

        Parameters
        ----------
        current_worker_key : str
            The key of the worker that needs argument injection.
        current_automa : GraphAutoma
            The automa instance containing the worker graph and context.

        Returns
        -------
        Tuple[Tuple[Any, ...], Dict[str, Any]]
            A tuple containing (positional_args, keyword_args) with injected values
            for parameters marked with special descriptors.
        """
        current_worker_sig = self._worker_rule_dict[current_worker_key]["param_names"]
        return self._injector.inject(current_worker_key, current_worker_sig, current_automa)

    def update_data_flow_topology(
        self, 
        dynamic_tasks: List[Union["_AddWorkerDeferredTask", "_RemoveWorkerDeferredTask", "_AddDependencyDeferredTask"]],
    ) -> None:
        """
        Update the data flow topology based on dynamic graph modifications.

        This method processes deferred tasks that modify the worker graph at runtime,
        updating internal data structures to reflect changes such as adding workers,
        removing workers, or adding dependencies.

        Parameters
        ----------
        dynamic_tasks : List[Union["_AddWorkerDeferredTask", "_RemoveWorkerDeferredTask", "_AddDependencyDeferredTask"]]
            List of deferred tasks representing graph modifications to apply.
            Each task can be one of:
            - AddWorker: Add a new worker to the graph
            - RemoveWorker: Remove a worker from the graph
            - AddDependency: Add a dependency relationship between workers
        """
        for dynamic_task in dynamic_tasks:
            if dynamic_task.task_type == "add_worker":
                key = dynamic_task.worker_key
                worker_obj = dynamic_task.worker_obj
                dependencies: List[str] = deepcopy(dynamic_task.dependencies)
                is_start: bool = dynamic_task.is_start
                args_mapping_rule: ArgsMappingRule = dynamic_task.args_mapping_rule
                result_dispatching_rule: ResultDispatchingRule = dynamic_task.result_dispatching_rule
                
                # update the _worker_forward_count according to the "add_worker" interface
                for trigger in dependencies:
                    if trigger not in self._worker_forward_count:
                        self._worker_forward_count[trigger] = {
                            "forward_count": 0,
                            "forward_index": 0,
                        }
                    self._worker_forward_count[trigger]["forward_count"] += 1

                # update the _worker_rule_dict according to the "add_worker" interface
                receiver_rule, sender_rule = args_mapping_rule, result_dispatching_rule
                dependencies = deepcopy(dependencies)  # Make sure do not affect the original worker's dependencies.
                if is_start:
                    dependencies.append("__automa__")
                self._worker_rule_dict[key] = {
                    "param_names": worker_obj.get_input_param_names(),
                    "dependencies": dependencies,
                    "receiver_rule": receiver_rule,
                    "sender_rule": sender_rule,
                }

            elif dynamic_task.task_type == "remove_worker":
                key = dynamic_task.worker_key
                dependencies = self._worker_rule_dict[key]["dependencies"]

                # update the _worker_forward_count according to the "remove_worker" interface
                for trigger in dependencies:
                    if trigger not in self._worker_forward_count:
                        continue
                    self._worker_forward_count[trigger]["forward_count"] -= 1

                # update the _worker_rule_dict according to the "remove_worker" interface
                for _, rule in self._worker_rule_dict.items():
                    if key in rule["dependencies"]:
                        rule["dependencies"].remove(key)
                del self._worker_rule_dict[key]
            
            elif dynamic_task.task_type == "add_dependency":
                key = dynamic_task.worker_key
                dependency = dynamic_task.dependency

                # update the _worker_forward_count according to the "add_dependency" interface
                if dependency not in self._worker_forward_count:
                    self._worker_forward_count[dependency] = {
                        "forward_count": 0,
                        "forward_index": 0
                    }
                self._worker_forward_count[dependency]["forward_count"] += 1

                # update the _worker_rule_dict according to the "add_dependency" interface
                self._worker_rule_dict[key]["dependencies"].append(dependency)
            

def safely_map_args(
    in_args: Tuple[Any, ...], 
    in_kwargs: Dict[str, Any],
    rx_param_names_dict: Dict[enum.IntEnum, List[str]],
) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    """
    Safely map input arguments to match a target function's parameter signature.

    This function filters and adjusts positional and keyword arguments to ensure they
    can be safely passed to a target function. It handles conflicts between positional
    and keyword arguments, and filters out invalid keyword arguments when the target
    function doesn't accept **kwargs.

    The mapping follows Python's argument binding rules:
    - Positional arguments take priority over keyword arguments
    - If a parameter is filled by a positional argument, any keyword argument with
      the same name will be ignored
    - POSITIONAL_ONLY parameters can only be filled by positional arguments
    - KEYWORD_ONLY parameters can only be filled by keyword arguments

    Parameters
    ----------
    in_args : Tuple[Any, ...]
        Input positional arguments to be mapped.
    in_kwargs : Dict[str, Any]
        Input keyword arguments to be mapped.
    rx_param_names_dict : Dict[_ParameterKind, List[str]]
        The parameter names dictionary of the receiver worker.

    Returns
    -------
    Tuple[Tuple[Any, ...], Dict[str, Any]]
        Mapped positional and keyword arguments that can be safely passed to the target function.
    """
    
    # Step 1: Extract function parameter information
    positional_only_param_names = [name for name, _ in rx_param_names_dict.get(Parameter.POSITIONAL_ONLY, [])]
    positional_or_keyword_param_names = [name for name, _ in rx_param_names_dict.get(Parameter.POSITIONAL_OR_KEYWORD, [])]
    keyword_only_param_names = [name for name, _ in rx_param_names_dict.get(Parameter.KEYWORD_ONLY, [])]
    var_positional_param_names = [name for name, _ in rx_param_names_dict.get(Parameter.VAR_POSITIONAL, [])]
    var_keyword_param_names = [name for name, _ in rx_param_names_dict.get(Parameter.VAR_KEYWORD, [])]

    # Step 2: Handle self/cls parameter for bound methods
    # For bound methods, the first POSITIONAL_OR_KEYWORD parameter is typically
    # 'self' or 'cls', which is already bound at call time. Therefore, positional
    # arguments should be mapped starting from the second parameter.
    skip_first_param = (
        positional_or_keyword_param_names 
        and positional_or_keyword_param_names[0] in ('self', 'cls')
    )
    
    # Step 3: Calculate positional argument binding
    # Python's argument binding rule: positional arguments fill parameters in order.
    # Binding order: POSITIONAL_ONLY first, then POSITIONAL_OR_KEYWORD (skipping self/cls)
    # 
    # 3.1 Calculate how many positional arguments fill POSITIONAL_ONLY parameters
    num_pos_only_filled = min(len(in_args), len(positional_only_param_names))
    remaining_pos_args = max(0, len(in_args) - num_pos_only_filled)
    
    # 3.2 Calculate effective POSITIONAL_OR_KEYWORD parameters (excluding self/cls)
    effective_pos_or_kw = (
        positional_or_keyword_param_names[1:] if skip_first_param 
        else positional_or_keyword_param_names
    )
    # 3.3 Calculate how many positional arguments fill POSITIONAL_OR_KEYWORD parameters
    num_pos_or_kw_filled = min(remaining_pos_args, len(effective_pos_or_kw))
    
    # Step 4: Identify parameters filled by positional arguments
    # Record the set of POSITIONAL_OR_KEYWORD parameter names that are filled by
    # positional arguments. These parameters must be removed from kwargs to avoid
    # conflicts.
    param_start_idx = 1 if skip_first_param else 0
    filled_pos_or_kw_params = set(
        positional_or_keyword_param_names[param_start_idx:param_start_idx + num_pos_or_kw_filled]
    )
    
    # Step 5: Filter conflicting keyword arguments
    # Core principle: positional arguments take precedence over keyword arguments.
    # If a parameter is already filled by a positional argument, the corresponding
    # keyword argument in kwargs must be removed to avoid "got multiple values
    # for argument" errors.
    rx_kwargs = {k: v for k, v in in_kwargs.items() if k not in filled_pos_or_kw_params}
    
    # Step 6: Filter invalid keyword arguments
    # If the function doesn't accept **kwargs, filter out keyword arguments that
    # are not in the function signature. Only allow:
    # - KEYWORD_ONLY parameters (e.g., g, h)
    # - Unfilled POSITIONAL_OR_KEYWORD parameters
    if not var_keyword_param_names:
        allowed_kw_params = set(keyword_only_param_names)
        unfilled_pos_or_kw_params = set(
            positional_or_keyword_param_names[param_start_idx + num_pos_or_kw_filled:]
        )
        allowed_kw_params.update(unfilled_pos_or_kw_params)
        rx_kwargs = {k: v for k, v in rx_kwargs.items() if k in allowed_kw_params}
    
    # Step 7: Handle special cases
    # When a predecessor worker returns None and the successor worker accepts no
    # arguments, return empty arguments.
    total_fixed_params = len(positional_only_param_names) + len(effective_pos_or_kw)
    if len(in_args) == 1 and in_args[0] is None and total_fixed_params == 0 and not var_positional_param_names:
        return (), {}
    
    return in_args, rx_kwargs


def get_filled_params(
    final_args: Tuple[Any, ...], 
    final_kwargs: Dict[str, Any],
    param_names: List[Tuple[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """
    Map parameter names to their argument sources and values.
    
    Returns a dictionary mapping each parameter name to a dictionary containing:
    - 'source': 'args' or 'kwargs' indicating where the argument comes from
    - 'index': int (if from args) - the index in final_args
    - 'key': str (if from kwargs) - the key name in final_kwargs
    - 'value': the actual argument value
    
    Parameters
    ----------
    final_args : Tuple[Any, ...]
        Final positional arguments to be analyzed.
    final_kwargs : Dict[str, Any]
        Final keyword arguments to be analyzed.
    param_names : List[Tuple[str, Any]]
        The parameter names to be analyzed.
    
    Returns
    -------
    Dict[str, Dict[str, Any]]
        A dictionary mapping parameter names to their argument source information.
        Each value is a dict with keys: 'source', 'index' (optional), 'key' (optional), 'value'
        
    Examples
    --------
    >>> # Function: func(self, x, y=2, *, z=3)
    >>> # Input: final_args = (1,), final_kwargs = {'y': 5, 'z': 10}
    >>> result = get_filled_params((1,), {'y': 5, 'z': 10}, worker)
    >>> # Result: {
    >>> #     'x': {'source': 'args', 'index': 0, 'value': 1},
    >>> #     'y': {'source': 'kwargs', 'key': 'y', 'value': 5},
    >>> #     'z': {'source': 'kwargs', 'key': 'z', 'value': 10}
    >>> # }
    """
    positional_only_param_names = [name for name, _ in param_names.get(Parameter.POSITIONAL_ONLY, [])]
    positional_or_keyword_param_names = [name for name, _ in param_names.get(Parameter.POSITIONAL_OR_KEYWORD, [])]
    keyword_only_param_names = [name for name, _ in param_names.get(Parameter.KEYWORD_ONLY, [])]
    
    # Handle self/cls parameter for bound methods
    skip_first_param = (
        positional_or_keyword_param_names 
        and positional_or_keyword_param_names[0] in ('self', 'cls')
    )
    
    # Calculate positional argument binding
    num_pos_only_filled = min(len(final_args), len(positional_only_param_names))
    remaining_pos_args = max(0, len(final_args) - num_pos_only_filled)
    
    effective_pos_or_kw = (
        positional_or_keyword_param_names[1:] if skip_first_param 
        else positional_or_keyword_param_names
    )
    num_pos_or_kw_filled = min(remaining_pos_args, len(effective_pos_or_kw))
    
    result = {}
    param_start_idx = 1 if skip_first_param else 0
    
    # Map POSITIONAL_ONLY parameters filled by positional args
    for i, param_name in enumerate(positional_only_param_names[:num_pos_only_filled]):
        result[param_name] = {
            'source': 'args',
            'index': i,
            'value': final_args[i]
        }
    
    # Map POSITIONAL_OR_KEYWORD parameters filled by positional args
    for i, param_name in enumerate(
        positional_or_keyword_param_names[param_start_idx:param_start_idx + num_pos_or_kw_filled]
    ):
        arg_index = num_pos_only_filled + i
        result[param_name] = {
            'source': 'args',
            'index': arg_index,
            'value': final_args[arg_index]
        }
    
    # Map parameters filled by keyword arguments (only if not already filled by positional)
    for param_name in keyword_only_param_names:
        if param_name in final_kwargs:
            result[param_name] = {
                'source': 'kwargs',
                'key': param_name,
                'value': final_kwargs[param_name]
            }
    
    for param_name in positional_or_keyword_param_names[param_start_idx:]:
        if param_name in final_kwargs and param_name not in result:
            result[param_name] = {
                'source': 'kwargs',
                'key': param_name,
                'value': final_kwargs[param_name]
            }

    return result


def override_func_signature(
    name: str,
    func: Union[Callable, MethodType],
    data: Dict[enum.IntEnum, List[Tuple[str, Any]]],
) -> None:
    """
    Override the signature of a function or method, updating only parameters present in data.
    
    This function preserves the original parameter order and keeps VAR_POSITIONAL and VAR_KEYWORD
    parameters unchanged. It only updates parameters that exist in both the original signature
    and the provided data.
    
    Parameters
    ----------
    name : str
        The name of the function (used for error messages).
    func : Union[Callable, MethodType]
        The function or method to override the signature for.
    data : Dict[enum.IntEnum, List[Tuple[str, Any]]]
        Parameter data from get_param_names_all_kinds, where key is Parameter kind,
        value is list of (param_name, default_value) tuples.
    """
    if isinstance(func, MethodType):
        func = func.__func__
    original_sig = inspect.signature(func)
    original_params_dict = {p.name: p for p in original_sig.parameters.values()}
    
    # Build a mapping of param_name -> (param_kind, default_value) from data
    data_params_map = {}
    for param_kind, param_list in data.items():
        for param_name, default_value in param_list:
            if param_name in data_params_map:
                raise ValueError(f"Duplicate parameter '{param_name}' in data")
            data_params_map[param_name] = (param_kind, default_value)
    
    # Validate: only allow overriding existing non-varargs names
    existing_param_names = set(original_params_dict.keys())
    extra_keys = set(data_params_map.keys()) - existing_param_names
    if extra_keys:
        raise TypeError(f"{name} has unsupported parameters: {sorted(extra_keys)}")
    
    # Override the function signature (preserve original order and VAR_POSITIONAL/VAR_KEYWORD)
    new_params = []
    for param_name, original_param in original_sig.parameters.items():
        if original_param.kind in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD):
            # Keep VAR_POSITIONAL and VAR_KEYWORD unchanged
            new_params.append(original_param)
            continue
        
        if param_name in data_params_map:
            # Update this parameter from data
            param_kind, default_value = data_params_map[param_name]
            
            # Convert inspect._empty to Parameter.empty for consistency
            if default_value is inspect._empty:
                default_value = Parameter.empty
            
            # Preserve original annotation if available
            annotation = original_param.annotation
            
            new_param = Parameter(
                name=param_name,
                kind=param_kind,
                default=default_value,
                annotation=annotation,
            )
            new_params.append(new_param)
        else:
            # Keep original parameter unchanged
            new_params.append(original_param)
    
    new_signature = original_sig.replace(parameters=new_params)
    setattr(func, "__signature__", new_signature)


def set_func_signature(
    func: Callable,
    data: Dict[enum.IntEnum, List[Tuple[str, Any]]],
) -> None:
    """
    Set the signature of a function based on provided data from get_param_names_all_kinds.
    
    Parameters
    ----------
    func : Callable
        The function to set the signature for.
    data : Dict[enum.IntEnum, List[Tuple[str, Any]]]
        Parameter data from get_param_names_all_kinds, where key is Parameter kind,
        value is list of (param_name, default_value) tuples.
    """
    # Get original signature to extract type annotations
    original_sig = inspect.signature(func)
    original_params_dict = {p.name: p for p in original_sig.parameters.values()}
    
    # Process parameters in the correct order: POSITIONAL_ONLY -> POSITIONAL_OR_KEYWORD -> VAR_POSITIONAL -> KEYWORD_ONLY -> VAR_KEYWORD
    param_kinds_order = [
        Parameter.POSITIONAL_ONLY,
        Parameter.POSITIONAL_OR_KEYWORD,
        Parameter.VAR_POSITIONAL,
        Parameter.KEYWORD_ONLY,
        Parameter.VAR_KEYWORD,
    ]
    
    params = []
    for param_kind in param_kinds_order:
        if param_kind not in data:
            continue
        
        param_list = data[param_kind]
        for param_name, default_value in param_list:
            # Get annotation from original signature if available
            annotation = inspect._empty
            if param_name in original_params_dict:
                annotation = original_params_dict[param_name].annotation
            
            # Convert inspect._empty to Parameter.empty if needed
            # (they are the same object, but we use Parameter.empty for consistency)
            if default_value is inspect._empty:
                default_value = Parameter.empty
            
            param = Parameter(
                name=param_name,
                kind=param_kind,
                default=default_value,
                annotation=annotation,
            )
            params.append(param)
    
    new_signature = inspect.Signature(parameters=params)
    setattr(func, "__signature__", new_signature)


def set_method_signature(
    method: MethodType,
    data: Dict[enum.IntEnum, List[Tuple[str, Any]]],
) -> None:
    """
    Set the signature of a method based on provided data from get_param_names_all_kinds.
    
    Parameters
    ----------
    method : MethodType
        The bound method to set the signature for.
    data : Dict[enum.IntEnum, List[Tuple[str, Any]]]
        Parameter data from get_param_names_all_kinds, where key is Parameter kind,
        value is list of (param_name, default_value) tuples.
    """
    # Get the original signature to preserve 'self' parameter
    original_sig = inspect.signature(method.__func__)
    original_params = list(original_sig.parameters.values())
    original_params_dict = {p.name: p for p in original_params}
    
    # Store the original signature on the class if not already stored
    # This allows us to restore it for instances that don't have custom signatures
    func = method.__func__
    original_sig_attr = f"__{func.__name__}_original_signature__"
    if not hasattr(func, original_sig_attr):
        setattr(func, original_sig_attr, original_sig)
    
    # Extract 'self' parameter if it exists (should be the first parameter for instance methods)
    self_param = None
    if original_params and original_params[0].name == 'self':
        self_param = original_params[0]
    
    # Process parameters in the correct order: POSITIONAL_ONLY -> POSITIONAL_OR_KEYWORD -> VAR_POSITIONAL -> KEYWORD_ONLY -> VAR_KEYWORD
    param_kinds_order = [
        Parameter.POSITIONAL_ONLY,
        Parameter.POSITIONAL_OR_KEYWORD,
        Parameter.VAR_POSITIONAL,
        Parameter.KEYWORD_ONLY,
        Parameter.VAR_KEYWORD,
    ]
    
    params = []
    for param_kind in param_kinds_order:
        if param_kind not in data:
            continue
        
        param_list = data[param_kind]
        for param_name, default_value in param_list:
            # Skip 'self' parameter as it will be added separately
            if param_name == 'self':
                continue
            
            # Get annotation from original signature if available
            annotation = inspect._empty
            if param_name in original_params_dict:
                annotation = original_params_dict[param_name].annotation
            
            # Convert inspect._empty to Parameter.empty if needed
            # (they are the same object, but we use Parameter.empty for consistency)
            if default_value is inspect._empty:
                default_value = Parameter.empty
            
            param = Parameter(
                name=param_name,
                kind=param_kind,
                default=default_value,
                annotation=annotation,
            )
            params.append(param)
    
    # Combine: self (if exists) + params
    final_params = []
    if self_param is not None:
        final_params.append(self_param)
    final_params.extend(params)
    
    new_signature = inspect.Signature(parameters=final_params)
    
    # For bound methods, set signature on the underlying function object
    # Bound method objects are read-only, so we need to modify __func__.__signature__
    # This maintains backward compatibility for code that reads from the class method
    setattr(method.__func__, "__signature__", new_signature)
    
    # Additionally, store the signature on the instance to allow per-instance signatures
    # This prevents signature pollution between instances when set_method_signature is called
    # The instance signature takes precedence in get_param_names_all_kinds
    instance = method.__self__
    method_name = method.__func__.__name__
    signature_attr = f"__{method_name}_signature__"
    setattr(instance, signature_attr, new_signature)