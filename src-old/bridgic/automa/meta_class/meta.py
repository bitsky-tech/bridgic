import inspect
from inspect import Signature
from typing import Any
from pydantic import create_model, ConfigDict
from bridgic.automa.constrained_states import _OutputValueDescriptor

class AutoMaMeta(type):
        def __new__(mcls, name, bases, dict):
            cls = super().__new__(mcls, name, bases, dict)

            worker_bridges = []
            func_return_types: dict[str, _OutputValueDescriptor] = {}
            for name, attr_value in dict.items():
                  if hasattr(attr_value, "_bridge_info"):
                        func_signature = inspect.signature(attr_value._bridge_info.func)
                        if func_signature.return_annotation != Signature.empty:
                              func_return_type = func_signature.return_annotation
                        else:
                              func_return_type = Any
                        default_return_val = _OutputValueDescriptor(value_type=func_return_type)
                        func_return_types[name] = default_return_val
                        worker_bridges.append(attr_value._bridge_info)

              # TODO: 检测有没有end节点
            if len(worker_bridges) > 0:
                  setattr(cls, "_worker_bridges", worker_bridges)
            if len(func_return_types) > 0:
                  setattr(cls, "_output_buffer", func_return_types)
            return cls
