from inspect import Parameter
from bridgic.utils.inspect_tools import load_qualified_class_or_func, get_param_names_by_kind

class A:
    def func_1(self, a: int, b: str, c=5, d=6) -> int:
        pass

    def func_2(self, a: int, /, b: str, c=5, d=6) -> int:
        pass

    def func_3(self, a: int, /, b: str, c=5, d=6, *, e: str, f:int=7, g, h="", **kwargs) -> int:
        pass

    def func_4(self, a: int, /, b: str, c=5, d=6, *args, **kwargs) -> int:
        pass

def test_get_arg_names_without_defaults():
    a = A()

    assert get_param_names_by_kind(a.func_1, Parameter.POSITIONAL_ONLY) == []
    assert get_param_names_by_kind(a.func_1, Parameter.POSITIONAL_OR_KEYWORD) == ["a", "b", "c", "d"]
    assert get_param_names_by_kind(a.func_1, Parameter.POSITIONAL_OR_KEYWORD, exclude_default=True) == ["a", "b"]

    assert get_param_names_by_kind(a.func_2, Parameter.POSITIONAL_ONLY) == ["a"]
    assert get_param_names_by_kind(a.func_2, Parameter.POSITIONAL_ONLY, exclude_default=True) == ["a"]
    assert get_param_names_by_kind(a.func_2, Parameter.POSITIONAL_OR_KEYWORD) == ["b", "c", "d"]
    assert get_param_names_by_kind(a.func_2, Parameter.POSITIONAL_OR_KEYWORD, exclude_default=True) == ["b"]

    assert get_param_names_by_kind(a.func_3, Parameter.POSITIONAL_ONLY) == ["a"]
    assert get_param_names_by_kind(a.func_3, Parameter.POSITIONAL_ONLY, exclude_default=True) == ["a"]
    assert get_param_names_by_kind(a.func_3, Parameter.POSITIONAL_OR_KEYWORD) == ["b", "c", "d"]
    assert get_param_names_by_kind(a.func_3, Parameter.POSITIONAL_OR_KEYWORD, exclude_default=True) == ["b"]
    assert get_param_names_by_kind(a.func_3, Parameter.KEYWORD_ONLY) == ["e", "f", "g", "h"]
    assert get_param_names_by_kind(a.func_3, Parameter.KEYWORD_ONLY, exclude_default=True) == ["e", "g"]
    assert get_param_names_by_kind(a.func_3, Parameter.VAR_KEYWORD) == ["kwargs"]

    assert get_param_names_by_kind(a.func_4, Parameter.POSITIONAL_ONLY) == ["a"]
    assert get_param_names_by_kind(a.func_4, Parameter.POSITIONAL_ONLY, exclude_default=True) == ["a"]
    assert get_param_names_by_kind(a.func_4, Parameter.POSITIONAL_OR_KEYWORD) == ["b", "c", "d"]
    assert get_param_names_by_kind(a.func_4, Parameter.POSITIONAL_OR_KEYWORD, exclude_default=True) == ["b"]
    assert get_param_names_by_kind(a.func_4, Parameter.VAR_POSITIONAL) == ["args"]
    assert get_param_names_by_kind(a.func_4, Parameter.VAR_KEYWORD) == ["kwargs"]

class C:
    class D:
        class E:
            def f1(self):
                pass
            async def f2_async(self):
                pass

def test_load_qualified_class():
    from bridgic.automa.worker import Worker
    qualified_name = Worker.__module__ + "." + Worker.__qualname__
    # "bridgic.automa.worker.worker.Worker"
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
    # "bridgic.utils.inspect_tools.load_qualified_class"
    func = load_qualified_class_or_func(func_qualified_name)
    assert func is load_qualified_class_or_func