from contextvars import ContextVar

graph_stack = ContextVar("graph_stack", default=[])


class MyObj:
    def __init__(self, name):
        self.name = name
        self.other_name = None

    def __rshift__(self, other):
        print(f"{self.name} __rshift__ {other.name}")
        self.other_name = other.name
        return self


class graph:
    def __init__(self, name=None):
        self.name = name
        self.registry = {}

    def __enter__(self):
        stack = list(graph_stack.get())
        stack.append(self)
        graph_stack.set(stack)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        stack = list(graph_stack.get())
        stack.pop()
        graph_stack.set(stack)

    def register(self, name, value):
        print(f"[{self.name or 'graph'}] register {name}")
        self.registry[name] = value


class TrackingNamespace(dict):
    def __setitem__(self, key, value):
        # 只对特定类型的值进行 MyObj 包装，避免影响类名等特殊值
        if isinstance(value, str) and not key.startswith('__'):
            value = MyObj(name=value)
        super().__setitem__(key, value)

        stack = graph_stack.get()

        # 当前活跃的 graph（如果有）
        if not stack:
            return

        current_graph = stack[-1]

        # 如果是 graph 且正在 with 块中，自动命名并注册到父图
        if isinstance(value, graph):
            # 如果当前 value 还没进入 with（__enter__ 未调用），则先跳过
            if value not in stack:
                return

            # 自动命名
            if not value.name:
                value.name = key

            # 找父图
            if len(stack) >= 2 and stack[-1] is value:
                parent = stack[-2]
                parent.register(value.name, value)
            return

        # 普通对象，注册到当前图
        current_graph.register(key, value)


class ComponentMeta(type):
    @classmethod
    def __prepare__(mcls, name, bases):
        return TrackingNamespace()


class component(metaclass=ComponentMeta):
    pass


# ==== 测试 ====
class MyGraph(component):
    x: int = None

    with graph() as g:
        a = "worker1"

        with graph() as g2:
            d = "worker1"

        b = "worker2"

        a >> b

