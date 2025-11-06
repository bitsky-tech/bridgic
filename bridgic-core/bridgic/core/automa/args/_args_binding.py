from dataclasses import dataclass
from typing import Tuple, Any, Union, List


@dataclass
class GatherMode:
    data: Any


@dataclass
class DistributeMode:
    data: Union[Tuple[Any, ...], List[Any, ...]]

    def __post_init__(self) -> None:
        if not isinstance(self.data, (Tuple, List)):
            raise ValueError(f"The distributed data must be a tuple or a list, not {type(self.data)}")


@dataclass
class MergeMode:
    params: Any


@dataclass
class AsIsMode:
    params: Any


class ArgsManager:
    """
    A class for binding arguments to a worker.
    """
    def __init__(
        self,
        data_to_send: Tuple[Union[GatherMode, DistributeMode]],
        params_to_receive: Tuple[Union[MergeMode, AsIsMode]],
    ) -> None:
        pass




