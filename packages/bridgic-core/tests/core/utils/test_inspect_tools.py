import inspect
from inspect import Parameter
from typing import Annotated

from bridgic.core.utils._console import printer
from bridgic.core.utils._inspect_tools import (
    load_qualified_class_or_func,
    get_param_names_by_kind,
    get_param_names_all_kinds,
    get_tool_description_from,
)

class A:
    def func_1(self, a: int, b: str, c=5, d=6) -> int:
        pass

    def func_2(self, a: int, /, b: str, c=5, d=6) -> int:
        pass

    def func_3(self, a: int, /, b: str, c=5, d=6, *, e: str, f:int=7, g, h="", **kwargs) -> int:
        pass

    def func_4(self, a: int, /, b: str, c=5, d=6, *args, e=7, f, g, **kwargs) -> int:
        pass

def test_get_param_names_by_kind():
    a = A()

    assert get_param_names_by_kind(a.func_1, Parameter.POSITIONAL_ONLY) == []
    assert get_param_names_by_kind(a.func_1, Parameter.POSITIONAL_OR_KEYWORD) == ["self", "a", "b", "c", "d"]
    assert get_param_names_by_kind(a.func_1, Parameter.POSITIONAL_OR_KEYWORD, exclude_default=True) == ["self", "a", "b"]

    assert get_param_names_by_kind(a.func_2, Parameter.POSITIONAL_ONLY) == ["self", "a"]
    assert get_param_names_by_kind(a.func_2, Parameter.POSITIONAL_OR_KEYWORD) == ["b", "c", "d"]
    assert get_param_names_by_kind(a.func_2, Parameter.POSITIONAL_OR_KEYWORD, exclude_default=True) == ["b"]

    assert get_param_names_by_kind(a.func_3, Parameter.POSITIONAL_ONLY) == ["self", "a"]
    assert get_param_names_by_kind(a.func_3, Parameter.POSITIONAL_OR_KEYWORD) == ["b", "c", "d"]
    assert get_param_names_by_kind(a.func_3, Parameter.POSITIONAL_OR_KEYWORD, exclude_default=True) == ["b"]
    assert get_param_names_by_kind(a.func_3, Parameter.KEYWORD_ONLY) == ["e", "f", "g", "h"]
    assert get_param_names_by_kind(a.func_3, Parameter.KEYWORD_ONLY, exclude_default=True) == ["e", "g"]
    assert get_param_names_by_kind(a.func_3, Parameter.VAR_KEYWORD) == ["kwargs"]

    assert get_param_names_by_kind(a.func_4, Parameter.POSITIONAL_ONLY) == ["self", "a"]
    assert get_param_names_by_kind(a.func_4, Parameter.POSITIONAL_OR_KEYWORD) == ["b", "c", "d"]
    assert get_param_names_by_kind(a.func_4, Parameter.POSITIONAL_OR_KEYWORD, exclude_default=True) == ["b"]
    assert get_param_names_by_kind(a.func_4, Parameter.KEYWORD_ONLY) == ["e", "f", "g"]
    assert get_param_names_by_kind(a.func_4, Parameter.KEYWORD_ONLY, exclude_default=True) == ["f", "g"]
    assert get_param_names_by_kind(a.func_4, Parameter.VAR_POSITIONAL) == ["args"]
    assert get_param_names_by_kind(a.func_4, Parameter.VAR_KEYWORD) == ["kwargs"]

def test_get_param_names_all_kinds():
    a = A()
    
    assert get_param_names_all_kinds(a.func_1) == {
        Parameter.POSITIONAL_OR_KEYWORD: [('self', inspect._empty), ("a", inspect._empty), ("b", inspect._empty), ("c", 5), ("d", 6)],
    }
    assert get_param_names_all_kinds(a.func_2) == {
        Parameter.POSITIONAL_ONLY: [('self', inspect._empty), ("a", inspect._empty)],
        Parameter.POSITIONAL_OR_KEYWORD: [("b", inspect._empty), ("c", 5), ("d", 6)],
    }
    assert get_param_names_all_kinds(a.func_3) == {
        Parameter.POSITIONAL_ONLY: [('self', inspect._empty), ("a", inspect._empty)],
        Parameter.POSITIONAL_OR_KEYWORD: [("b", inspect._empty), ("c", 5), ("d", 6)],
        Parameter.KEYWORD_ONLY: [("e", inspect._empty), ("f", 7), ("g", inspect._empty), ("h", "")],
        Parameter.VAR_KEYWORD: [("kwargs", inspect._empty)],
    }
    assert get_param_names_all_kinds(a.func_4) == {
        Parameter.POSITIONAL_ONLY: [('self', inspect._empty), ("a", inspect._empty)],
        Parameter.POSITIONAL_OR_KEYWORD: [("b", inspect._empty), ("c", 5), ("d", 6)],
        Parameter.KEYWORD_ONLY: [("e", 7), ("f", inspect._empty), ("g", inspect._empty)],
        Parameter.VAR_POSITIONAL: [("args", inspect._empty)],
        Parameter.VAR_KEYWORD: [("kwargs", inspect._empty)],
    }

class C:
    class D:
        class E:
            def f1(self):
                pass
            async def f2_async(self):
                pass

def test_load_qualified_class():
    from bridgic.core.automa.worker import Worker
    qualified_name = Worker.__module__ + "." + Worker.__qualname__
    cls = load_qualified_class_or_func(qualified_name)
    assert cls is Worker

    qualified_name = C.D.E.__module__ + "." + C.D.E.__qualname__
    # "test_inspect_tools.C.D.E"
    cls = load_qualified_class_or_func(qualified_name)
    assert cls is C.D.E

def test_load_functions_or_methods():
    # Test loading a normal class function.
    f1 = C.D.E.f1
    f1_qualified_name = f1.__module__ + "." + f1.__qualname__
    # "test_inspect_tools.C.D.E.f1"
    f1_func = load_qualified_class_or_func(f1_qualified_name)
    assert f1_func is f1

    # Test loading an async class function.
    f2_async = C.D.E.f2_async
    f2_async_qualified_name = f2_async.__module__ + "." + f2_async.__qualname__
    # "test_inspect_tools.C.D.E.f2_async"
    f2_async_func = load_qualified_class_or_func(f2_async_qualified_name)
    assert f2_async_func is f2_async

    # Test loading a nomal function defined in a module.
    func_qualified_name = load_qualified_class_or_func.__module__ + "." + load_qualified_class_or_func.__qualname__
    func = load_qualified_class_or_func(func_qualified_name)
    assert func is load_qualified_class_or_func

def test_get_tool_description_from_docstring():
    def func(x: int) -> int:
        """
        Short summary of func.

        More details about func.

        Returns
        -------
        int
            The result of the function.
        """
        return x

    desc = get_tool_description_from(func)
    assert len(desc) > 0

def test_get_tool_description_from_signature():
    def func(a: int, b: Annotated[int, "meta"] = 1, *, c: str = "x") -> int:
        return a

    desc = get_tool_description_from(func)
    assert desc == "func(a: int, b: int, *, c: str) -> int\n"

def test_get_tool_description_ignores_self_and_cls():
    class C:
        def m(self, x: int):
            return x

        @classmethod
        def n(cls, x: int):
            return x

    desc_m = get_tool_description_from(C.m)
    assert desc_m == "m(x: int)\n"

    desc_n = get_tool_description_from(C.n)
    assert desc_n == "n(x: int)\n"
