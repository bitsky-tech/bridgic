import asyncio
from bridgic.core.automa import GraphAutoma, worker
from bridgic.core.agentic.asl import ASLAutoma, graph, concurrent, ASLField, Settings


async def produce_tasks(user_input: int) -> list:
        return [
            user_input + 1,
            user_input + 2,
            user_input + 3,
            user_input + 4,
            user_input + 5,
        ]

async def tasks_done(task_input: int) -> int:
    return task_input + 3


class MyASLGraph(ASLAutoma):
    with graph as sub_graph_5:  # input: 14 -> res: [18, 19, 20, 21, 22]
        a = produce_tasks  # [15, 16, 17, 18, 19]
        with concurrent(subtasks = ASLField(list, distribute=True)) as sub_concurrent:
            dynamic_logic = lambda subtasks, **kwargs: (
                tasks_done *Settings(
                    key=f"tasks_done_{i}"
                )
                for i, subtask in enumerate(subtasks)
            )
        
        +a >> ~sub_concurrent  # [18, 19, 20, 21, 22]


if __name__ == "__main__":
    my_asl_graph = MyASLGraph()
    res = asyncio.run(my_asl_graph.arun(user_input=1))
    print(res)
    