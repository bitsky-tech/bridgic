from typing import Any
from bridgic.utils.inspect_tools import get_arg_names

class A:
    def func_1(self, a: int, s: str, c, d, *args: Any, **kwargs) -> int:
        pass

    def func_2(self, a: int, s: str, /, c, e: str="e", f="f") -> None:
        pass

    def func_3(self, a, s, c):
        pass


def test_get_arg_names_without_defaults():
    a = A()
    assert get_arg_names(a.func_1) == ["a", "s", "c", "d", "args", "kwargs"]
    assert get_arg_names(a.func_2) == ["a", "s", "c", "e", "f"]
    assert get_arg_names(a.func_3) == ["a", "s", "c"]

