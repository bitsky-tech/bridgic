from pydantic import BaseModel, Field
from dataclasses import dataclass
import asyncio
from typing import List
from bridgic.asl.component import Component, graph, concurrent, Data, Settings
from bridgic.core.automa.worker import Worker
from bridgic.core.automa.args import System, From


async def produce_tasks(x: int):
    res = [i for i in range(x)]
    return res


async def tasks_done(y: int):
    res = y + 1
    return res


async def merge_tasks(tasks: List[int]):
    res = sum(tasks)
    return res


class MyGraph(Component):
    x: int = Field(description="The number of tasks to be produced")

    with graph() as g:
        a = produce_tasks
        
        with concurrent(subtasks=From("a")) as c:
            dynamic_logic = lambda subtasks: [
                tasks_done *Data(y=subtask) *Settings(key=f"tasks_done_{i}")
                for i, subtask in enumerate(subtasks)
            ]


        b = merge_tasks

        +a >> c >> ~b



if __name__ == "__main__":
    graph = MyGraph()
    print(graph)
    asyncio.run(graph.arun(x=10))