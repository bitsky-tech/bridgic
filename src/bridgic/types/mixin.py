import inspect
import uuid

from typing import List, Callable, TYPE_CHECKING
from bridgic.consts.args_mapping_rule import *

if TYPE_CHECKING:
    from bridgic.automa.automa import Automa

class WithKeyMixin:
    def __init__(self, key: str = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.key: str = key or f"autokey-{uuid.uuid4().hex[:8]}"

class AdaptableMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.core_worker: "Automa" = None

class LandableMixin:
    def __init__(
        self,
        dependencies: List[str] = [],
        is_start: bool = False,
        args_mapping_rule: str = ARGS_MAPPING_RULE_SUPPRESSED,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dependencies: List[str] = dependencies
        self.is_start: bool = is_start
        self.args_mapping_rule: str = args_mapping_rule

class CallableMixin:
    def __init__(self, func: Callable, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func: Callable = func
        self.is_coro: bool = inspect.iscoroutinefunction(func)
