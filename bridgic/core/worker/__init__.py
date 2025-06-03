# bridgic.core.processor这个包的作用：
# 定义核心的执行器概念： Processor。
# Processor是上层编排层操纵的基础单元。

from .worker import *
from .function_processor import *
from .method_processor import *

__all__ = ["Worker", "FunctionProcessor", "MethodProcessor"]
