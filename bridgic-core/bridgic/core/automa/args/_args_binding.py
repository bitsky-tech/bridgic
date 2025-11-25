import json
from copy import deepcopy
from inspect import Parameter
from dataclasses import dataclass
from typing import Literal, Tuple, Any, List, Dict, Mapping, Union
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
class IN_ORDER:
    """
    A descriptor to indicate that data should be distributed to multiple workers. 

    When is used to input arguments or worker with this class, the data will be distributed
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
    data: Union[List, Tuple]

    def __post_init__(self):
        if not isinstance(self.data, (list, tuple)):
            raise ValueError(f"The data must be a list or tuple, but got {type(self.data)}")


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
            if isinstance(value, IN_ORDER):
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
    
        def _start_args_binding(worker_receiver_rule: ArgsMappingRule) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
            data_mode_list = []
            for key, value in self._start_arguments.items():
                sender_rule = self._worker_rule_dict[key]["sender_rule"]
                data_mode_list.append({
                    'worker_key': key,
                    'data': value.data if isinstance(value, IN_ORDER) else value,
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
            next_args, next_kwargs = _start_args_binding(worker_receiver_rule)
        elif kickoff_single == "dependency":
            next_args, next_kwargs = _dependency_args_binding(worker_dependencies, worker_receiver_rule)
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

        propagation_kwargs = {}
        for key, value in input_kwargs.items():
            if key in positional_only_param_names:
                propagation_kwargs[key] = value
            elif key in positional_or_keyword_param_names:
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
            
