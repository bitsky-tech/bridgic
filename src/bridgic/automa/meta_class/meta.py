class AutoMaMeta(type):
        def __new__(mcls, name, bases, dict):
            cls = super().__new__(mcls, name, bases, dict)

            worker_bridges = []
            for name, attr_value in dict.items():
                  if hasattr(attr_value, "_bridge_info"):
                        worker_bridges.append(attr_value._bridge_info)

              # TODO: 检测有没有end节点
            setattr(cls, "_worker_bridges", worker_bridges)
            return cls
