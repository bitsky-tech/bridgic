from typing import List
import pytest

from bridgic.automa import GoapAutoma, worker, goal, DynamicOutputEffect
from pydantic import BaseModel

class SeachSubtask(BaseModel):
    aspect: str
    query: str

class SearchResult(BaseModel):
    title: str
    content: str
    from_url: str

class SubtaskingExample_SearchOrchestrator(GoapAutoma):
    @goal(priority=10) # Set this goal to ensure path planning before dynamic workers are created
    @worker(output_effects=DynamicOutputEffect.UnpackByType)
    def divide_search_subtask(self, user_input: str) -> List[SeachSubtask]:
        # TODO: call LLM to divide the search task into subtasks
        # Before dividing into subtasks, the exact number of subtasks is uncertain
        subtasks = [
            SeachSubtask(aspect="aspect1", query="query1"),
            SeachSubtask(aspect="aspect2", query="query2"),
            # ... more subtasks may exist
        ]
        # Spawn `dynamic workers` to execute the subtasks
        # The number of workers matches the number of subtasks
        for subtask in subtasks:
            self.add_func_as_worker(
                name=f"search_by_subtask_{subtask.aspect}",
                func=self.search_by_subtask,
                output_effects=DynamicOutputEffect.PackByType,
            )

        # Items in the subtasks list are subclasses of UnpackableItem, so they will be automatically unpacked and passed to dynamic workers
        return subtasks
    
    # Not annotated with @worker, intentionally; serves as dynamic workers.
    # Multiple instances will be created as needed.
    # Must use type hint (SeachSubtask) to specify the input parameter type in order to receive output marked as UnpackByType.
    def search_by_subtask(self, sub_task: SeachSubtask) -> SearchResult:
        # TODO: call search engine to search the results
        search_result = SearchResult(title="title", content="content", from_url="from_url")
        # Optional: dynamic workers can be safely removed here
        # self.remove_worker(f"search_by_subtask_{sub_task.aspect}")
        return search_result
    
    @goal(final=True)
    @worker(output_effects=["answer"])
    # Must use type hint (List[SearchResult]) to specify the input parameter type in order to receive outputs marked as PackByType.
    def synthesize_search_results(self, search_results: List[SearchResult]) -> str:
        # TODO: call LLM to synthesize the search results
        return "The answer is: xxx"
    
@pytest.fixture
def search_orchestrator():
    return SubtaskingExample_SearchOrchestrator()

@pytest.mark.asyncio
async def test_search_orchestrator(search_orchestrator):
    pass