from bridgic.core.automa.worker import CallableWorker
from bridgic.core.automa.args._args_binding import safely_map_args
from bridgic.core.utils._inspect_tools import get_param_names_all_kinds

# rx_args = (1, )
# rx_kwargs = {'x': 2}


# class MyWorker(CallableWorker):
#     def __init__(self):
#         super().__init__(self.func)

#     def func(self, x):
#         pass
    
# worker = MyWorker()
# args, kwargs = safely_map_args(rx_args, rx_kwargs, worker)
# worker.run(*args, **kwargs)


class A:
    def func_1(self, a: int, b: str, c=5, d=6) -> int:
        pass

    def func_2(self, a: int, /, b: str, c=5, d=6) -> int:
        pass

    def func_3(self, a: int, /, b: str, c=5, d=6, *, e: str, f:int=7, g, h="", **kwargs) -> int:
        pass

    def func_4(self, a: int, /, b: str, c=5, d=6, *args, e=7, f, g, **kwargs) -> int:
        pass

a = A()
print(get_param_names_all_kinds(a.func_1))
print()
print(get_param_names_all_kinds(a.func_2))
print()
print(get_param_names_all_kinds(a.func_3))
print()
print(get_param_names_all_kinds(a.func_4))