from typing import List
import pytest

from bridgic.automa import GraphAutoma, worker, goal, DynamicOutputEffect
from pydantic import BaseModel

class SeachSubtask(BaseModel):
    aspect: str
    query: str

class SearchResult(BaseModel):
    title: str
    content: str
    from_url: str

class SubtaskingExample_SearchOrchestrator(GraphAutoma):
    @worker(is_start=True)
    def divide_search_subtask(self, user_input: str) -> List[SeachSubtask]:
        # TODO: call LLM to divide the search task into subtasks.
        # Before dividing into subtasks, the exact number of subtasks is uncertain.
        # Here we hardcode the subtasks just for testing.
        subtasks = [
            SeachSubtask(aspect="aspect1", query="query1"),
            SeachSubtask(aspect="aspect2", query="query2"),
            SeachSubtask(aspect="aspect3", query="query3"),
            # ... more subtasks may exist
        ]
        # Spawn `dynamic workers` to execute the subtasks
        # The number of workers matches the number of subtasks
        for subtask in subtasks:
            new_worker_key = f"search_by_{subtask.aspect}"
            self.add_func_as_worker(
                key=new_worker_key,
                func=self.search_by_subtask,
            )
            self.ferry_to(new_worker_key, sub_task=subtask)

        # The return value is not used in this example.
        return subtasks

    # Not annotated with @worker, intentionally; serves as dynamic workers.
    # Multiple instances will be created as needed.
    def search_by_subtask(self, sub_task: SeachSubtask) -> SearchResult:
        # TODO: call search engine to search the results
        # Here we hardcode the search result just for testing.
        print(f"\n*********************************** in search_by_subtask(): sub_task={sub_task}\n")
        search_result = SearchResult(
            title=f"title_{sub_task.aspect}",
            content=f"content_{sub_task.aspect}",
            from_url=f"from_url_{sub_task.aspect}",
        )
        # Optional: dynamic workers can be safely removed here
        # self.remove_worker(f"search_by_{sub_task.aspect}")
        return search_result

    
    # Not annotated with @worker, intentionally; serves as dynamic workers.
    def synthesize_search_results(self, search_results: List[SearchResult]) -> str:
        # TODO: call LLM to synthesize the search results
        return "The answer is: xxx"
    
@pytest.fixture
def search_orchestrator():
    automa_obj = SubtaskingExample_SearchOrchestrator()
    automa_obj.set_running_options(debug=True)
    yield automa_obj

@pytest.mark.asyncio
async def test_search_orchestrator(search_orchestrator):
    await search_orchestrator.process_async(
        user_input="Please search for the latest news about Bridgic.",
    )
    