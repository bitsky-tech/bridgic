import asyncio
from bridgic.core.automa import GraphAutoma, worker, ArgsMappingRule

class MyGraphAutoma(GraphAutoma):
    @worker(is_start=True)
    async def greet(self) -> list[str]:
        return ["Hello", "Bridgic"]

    @worker(dependencies=["greet"], args_mapping_rule=ArgsMappingRule.AS_IS)
    async def output(self, message: list[str]):
        print(" ".join(message))

async def main():
    automa_obj = MyGraphAutoma(name="my_graph_automa", output_worker_key="output")
    await automa_obj.arun()

asyncio.run(main())