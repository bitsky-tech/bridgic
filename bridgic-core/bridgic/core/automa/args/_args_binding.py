import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, Tuple, Any, List, Dict, Mapping, Union
from typing_extensions import TYPE_CHECKING

from bridgic.core.types._common import ArgsMappingRule
from bridgic.core.types._error import WorkerArgsMappingError
from bridgic.core.utils._args_map import safely_map_args

if TYPE_CHECKING:
    from bridgic.core.automa._graph_automa import (
        _GraphAdaptedWorker,
        _AddWorkerDeferredTask,
        _RemoveWorkerDeferredTask,
        _AddDependencyDeferredTask,
    )

# type aliases
SENDER_RULES = Literal[ArgsMappingRule.GATHER, ArgsMappingRule.DISTRIBUTE]
RECEIVER_RULES = Literal[ArgsMappingRule.AS_IS, ArgsMappingRule.UNPACK, ArgsMappingRule.MERGE, ArgsMappingRule.SUPPRESSED]

# runtime type checking set
_SENDER_RULES_SET = {ArgsMappingRule.GATHER, ArgsMappingRule.DISTRIBUTE}
_RECEIVER_RULES_SET = {ArgsMappingRule.AS_IS, ArgsMappingRule.UNPACK, ArgsMappingRule.MERGE, ArgsMappingRule.SUPPRESSED}


@dataclass
class Distribute:
    data: Union[List, Tuple]

    def __post_init__(self):
        if not isinstance(self.data, (list, tuple)):
            raise ValueError(f"The data must be a list or tuple, but got {type(self.data)}")


def parse_args_mapping_rule(args_mapping_rule: ArgsMappingRule) -> Tuple[RECEIVER_RULES, SENDER_RULES]:
    """
    Parse the args_mapping_rule into a tuple of two modes, the first for the receiver and the second for the sender.

    Raises:
    -------
    ValueError: If the args_mapping_rule is not match any of the supported modes.
    ValueError: If the args_mapping_rule tuple has more than two elements.
    ValueError: If the two elements of the args_mapping_rule tuple are the same type.

    Returns:
    --------
    Tuple[RECEIVER_RULES, SENDER_RULES]: The parsed receiver and sender modes of the current worker.

    """
    def parse_single_mode(mode: ArgsMappingRule) -> Tuple[RECEIVER_RULES, SENDER_RULES]:
        # if the args_mapping_rule is a single mode, return the mode and the default mode.
        if mode in _RECEIVER_RULES_SET:
            return mode, ArgsMappingRule.GATHER
        elif mode in _SENDER_RULES_SET:
            return ArgsMappingRule.AS_IS, mode
        else:
            raise ValueError(f"The args_mapping_rule does not match any of the supported modes.")

    if isinstance(args_mapping_rule, Tuple):
        if len(args_mapping_rule) == 1:
            return parse_single_mode(args_mapping_rule[0])
        elif len(args_mapping_rule) == 2:
            rule_1, rule_2 = args_mapping_rule
            if type(rule_1) == type(rule_2):
                raise ValueError(f"The two elements of the args_mapping_rule tuple must be different, not {type(rule_1)}")
            if rule_1 in _SENDER_RULES_SET and rule_2 in _RECEIVER_RULES_SET:
                return rule_2, rule_1
            return rule_1, rule_2
        else:
            raise ValueError(f"The args_mapping_rule tuple must have 1 or 2 elements, not {len(args_mapping_rule)}")    
    else:
        return parse_single_mode(args_mapping_rule)


class ArgsManager:
    """
    A class for binding arguments to a worker.
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
        Initialize the ArgsManager.
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
            receiver_rule, sender_rule = parse_args_mapping_rule(worker.args_mapping_rule)
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
            if isinstance(value, Distribute):
                sender_rule = ArgsMappingRule.DISTRIBUTE
            else:
                sender_rule = ArgsMappingRule.GATHER
            self._worker_rule_dict[key] = {
                "dependencies": [],
                "param_names": [],
                "receiver_rule": None,
                "sender_rule": sender_rule,
            }

    ###############################################################################
    # Arguments Binding between workers that have dependency relationships.
    ###############################################################################
    def args_binding(
        self, 
        last_worker_key: str, 
        current_worker_key: str,
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        kickoff_single = "start" if last_worker_key == "__automa__" else "dependency"
        worker_dependencies = self._worker_rule_dict[current_worker_key]["dependencies"]
        worker_receiver_rule = self._worker_rule_dict[current_worker_key]["receiver_rule"]

        # If the last worker is not a dependency of the current worker, then return empty arguments.
        if not last_worker_key in worker_dependencies:
            return (), {}
    
        def _start_args_binding(worker_receiver_rule: RECEIVER_RULES) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
            data_mode_list = []
            for key, value in self._start_arguments.items():
                sender_rule = self._worker_rule_dict[key]["sender_rule"]
                data_mode_list.append({
                    'worker_key': key,
                    'data': value.data if isinstance(value, Distribute) else value,
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

        def _dependency_args_binding(worker_dependencies: List[str], worker_receiver_rule: RECEIVER_RULES) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
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

    ###############################################################################
    # Arguments Binding of Inputs Arguments.
    ###############################################################################
    def inputs_propagation(
        self,
        current_worker_key: str,
        next_args: Tuple[Any, ...],
        next_kwargs: Dict[str, Any],
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        input_kwargs = {k:v for k,v in self._input_kwargs.items() if k not in next_kwargs}
        next_kwargs = {**next_kwargs, **input_kwargs}
        rx_param_names_dict = self._worker_rule_dict[current_worker_key]["param_names"]
        next_args, next_kwargs = safely_map_args(next_args, next_kwargs, rx_param_names_dict)
        return next_args, next_kwargs

    def _args_send(self, data_mode_list: List[Dict[str, Any]]) -> List[Any]:
        """
        Send the data to the next workers.
        """
        send_data = []
        for data_mode in data_mode_list:
            worker_key = data_mode['worker_key']
            data = data_mode['data']
            send_rule = data_mode['send_rule']
            if send_rule == ArgsMappingRule.GATHER:
                send_data.append(data)
            elif send_rule == ArgsMappingRule.DISTRIBUTE:
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
        current_worker_receiver_rule: RECEIVER_RULES,
        data: List[Any]
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        """
        Resolve arguments mapping between workers that have dependency relationships.

        Parameters
        ----------
        last_worker_key : str
            The key of the last worker.
        current_worker_key : str
            The key of the current worker.
        current_worker_receiver_rule : RECEIVER_RULES
            The receiver rule of the worker.
        data : List[Any]
            The data to be mapped.
        
        Returns
        -------
        Tuple[Tuple[Any, ...], Dict[str, Any]]
            The mapped positional arguments and keyword arguments.
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

    def update_data_flow_topology(
        self, 
        dynamic_tasks: List[Union["_AddWorkerDeferredTask", "_RemoveWorkerDeferredTask", "_AddDependencyDeferredTask"]],
    ) -> None:
        """
        Update the data flow topology.
        """
        for dynamic_task in dynamic_tasks:
            if dynamic_task.task_type == "add_worker":
                key = dynamic_task.worker_key
                worker_obj = dynamic_task.worker_obj
                dependencies: List[str] = deepcopy(dynamic_task.dependencies)
                is_start: bool = dynamic_task.is_start
                args_mapping_rule: ArgsMappingRule = dynamic_task.args_mapping_rule
                
                # update the _worker_forward_count according to the "add_worker" interface
                for trigger in dependencies:
                    if trigger not in self._worker_forward_count:
                        self._worker_forward_count[trigger] = {
                            "forward_count": 0,
                            "forward_index": 0,
                        }
                    self._worker_forward_count[trigger]["forward_count"] += 1

                # update the _worker_rule_dict according to the "add_worker" interface
                receiver_rule, sender_rule = parse_args_mapping_rule(args_mapping_rule)
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
            
