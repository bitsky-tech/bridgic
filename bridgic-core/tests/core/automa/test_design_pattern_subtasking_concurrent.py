# from typing import List
# import pytest

# from bridgic.core.automa import GraphAutoma, worker, ArgsMappingRule
# from pydantic import BaseModel

# class SeachSubtask(BaseModel):
#     aspect: str
#     query: str

# class SearchResult(BaseModel):
#     title: str
#     content: str
#     from_url: str

# class SubtaskingExample_SearchOrchestrator(GraphAutoma):
#     @worker(is_start=True)
#     async def divide_search_subtask(self, user_input: str) -> List[SeachSubtask]:
#         # TODO: call LLM to divide the search task into subtasks.
#         # Before dividing into subtasks, the exact number of subtasks is uncertain.
#         # Here we hardcode the subtasks just for testing.
#         subtasks = [
#             SeachSubtask(aspect="aspect1", query="query1"),
#             SeachSubtask(aspect="aspect2", query="query2"),
#             SeachSubtask(aspect="aspect3", query="query3"),
#             # ... more subtasks may exist
#         ]
#         # Spawn `dynamic workers` to execute the subtasks
#         # The number of workers matches the number of subtasks
#         for subtask in subtasks:
#             new_worker_key = f"search_by_{subtask.aspect}"
#             self.add_func_as_worker(
#                 key=new_worker_key,
#                 func=self.search_by_subtask,
#             )
#             # Note: There is no way here to implement as sepecifying dependencies of these dynamic workers, because currently no args_mapping_rule is appropriate for this case.
#             # So we use ferry_to() to specify the arguments to the dynamic worker.
#             # TODO: Maybe need to add a new args_mapping_rule for this case.
#             self.ferry_to(new_worker_key, sub_task=subtask)
#         self.add_func_as_worker(
#             key="synthesize_search_results",
#             func=self.synthesize_search_results,
#             dependencies=[f"search_by_{subtask.aspect}" for subtask in subtasks],
#             args_mapping_rule=ArgsMappingRule.MERGE,
#         )
#         self.output_worker_key = "synthesize_search_results"

#         # The return value is not used in this example.
#         return subtasks

#     # Not annotated with @worker, intentionally; serves as dynamic workers.
#     # Multiple instances will be created as needed.
#     async def search_by_subtask(self, sub_task: SeachSubtask) -> SearchResult:
#         # TODO: call search engine to search the results
#         # Here we hardcode the search result just for testing.
#         search_result = SearchResult(
#             title=f"title_{sub_task.aspect}",
#             content=f"content_{sub_task.aspect}",
#             from_url=f"from_url_{sub_task.aspect}",
#         )
#         # Optional: dynamic workers can be safely removed here
#         # self.remove_worker(f"search_by_{sub_task.aspect}")
#         return search_result

    
#     # Not annotated with @worker, intentionally; serves as dynamic workers.
#     async def synthesize_search_results(self, search_results: List[SearchResult]) -> str:
#         # TODO: call LLM to synthesize the search results
#         return f"The answer is: content synthesized from {'#'.join([result.content for result in search_results])}"
    
# @pytest.fixture
# def search_orchestrator():
#     automa_obj = SubtaskingExample_SearchOrchestrator()
#     automa_obj.set_running_options(debug=True)
#     yield automa_obj

# @pytest.mark.asyncio
# async def test_search_orchestrator(search_orchestrator):
#     answer = await search_orchestrator.arun(
#         user_input="Please search for the latest news about Bridgic.",
#     )
#     assert answer == "The answer is: content synthesized from content_aspect1#content_aspect2#content_aspect3"
    