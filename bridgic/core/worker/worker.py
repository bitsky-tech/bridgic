from bridgic.core.worker.data_model import Task, TaskResult
from typing import Any, AsyncIterable

# Worker is the core element of the orchestration layer in the bridgic framework.
class Worker:
    # 输入参数及返回值的协议：
    # (输入参数或返回值的第一个元素，在需要的时候留作流式处理)
    # 输入参数或返回值的第一个元素，先看看是否是AsyncIterable；
    # 如果是，则剩余的元素按下面的规则<动态参数及返回值形式>处理；
    # 如果不是，则所有的元素按下面的规则<动态参数及返回值形式>处理。

    # 动态参数及返回值形式：

    # 输入参数的允许类型：
    # 1，单个参数，Task类型
    # 2，单个参数，其他任意类型
    # 3，多个参数，args
    # 4，多个参数，kwargs

    # 返回值的允许类型：
    # 1，单个返回值，TaskResult类型
    # 2，单个返回值，dict
    # 3，单个返回值，其他任意类型
    # 4，多个返回值，tuple
    async def process_async(self, *args, **kwargs) -> Any:
        if len(args) > 0 and isinstance(args[0], AsyncIterable):
            new_args = args[1:]

        # Validate arguments
        self._validate_args(*args, **kwargs)

        result = None
        if len(args) > 0:
            if isinstance(args[0], AsyncIterable):
                stream = args[0]
                result = await self.process_stream_async(stream, *new_args, **kwargs)
            elif isinstance(args[0], Task):
                task = args[0]
                result = await self.process_task_async(task)
            else:
                result = await self.process_default_async(*args, **kwargs)
        else:
            # By default, process_default_async handles other cases
            result = await self.process_default_async(*args, **kwargs)
        
        self._validate_result_values(result)

        return result
        
    async def process_task_async(self, task: Task) -> TaskResult:
        pass

    async def process_stream_async(self, stream: AsyncIterable, *args, **kwargs) -> Any:
        return stream

    async def process_default_async(self, *args, **kwargs) -> Any:
        pass

    def _validate_args(self, *args, **kwargs) -> None:
        if len(args) > 0:
            if len(kwargs) > 0:
                raise ValueError("args and kwargs are not allowed to be used together in a Worker")
            if isinstance(args[0], Task) and len(args) > 1:
                raise ValueError("If the first argument is a Task, additional arguments are not permitted.")            # 
            
    def _validate_result_values(self, result: Any) -> None:
        pass
