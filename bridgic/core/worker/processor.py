from bridgic.core.worker.data_model import ProcessorData, Task, InEvent, DataStream

# Worker is the core element of the orchestration layer in the bridgic framework.
# 分为同步和异步两类接口。
class Worker:

    # 默认都是异步接口
    async def process(self, data: ProcessorData) -> ProcessorData:
        # TODO: tracing
        if isinstance(data, Task):
            return await self.process_record(data)
        elif isinstance(data, InEvent):
            return await self.process_event(data)
        elif isinstance(data, DataStream):
            return await self.process_stream(data)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

    async def process_record(self, data: Task) -> ProcessorData:
        return data

    async def process_event(self, data: InEvent) -> ProcessorData:
        return data

    async def process_stream(self, data: DataStream) -> ProcessorData:
        return data
