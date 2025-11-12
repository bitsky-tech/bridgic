from bridgic.core.utils._inspect_tools import set_method_signature


class MyClass:
    def arun(self, x):
        return x + 1


params = {
    "y": {
        "type": int,
        "default": 1,
    }
}

my_class = MyClass()
set_method_signature(my_class.arun, params)
print(my_class.arun.__signature__)