import pytest
from bridgic.automa import GoapAutoma, worker, goal
from typing import List
from pydantic import BaseModel

class RoutingResult(BaseModel):
    sucess: bool # whether the routing is successful
    domain: str # The domain closest to user input

class Chunk(BaseModel):
    content: str
    from_url: str

class RoutingExample_RAGChatbot(GoapAutoma):

    @worker(output_effects=["routing_result"])
    def route_to_right_domain(self, user_input: str) -> RoutingResult:
        routing_result_1 = RoutingResult(sucess=True, domain="law")
        routing_result_2 = RoutingResult(sucess=True, domain="finance")
        # TODO: call LLM to route to the right domain
        return routing_result_1

    @worker(output_effects=["law_chunks"])
    def rag_in_law_domain(self, user_input: str, routing_result: RoutingResult) -> List[Chunk]:
        if not routing_result.sucess or routing_result.domain != "law":
            # TODO: return None or raise an Exception to stop here
            return None

        # TODO: query from RAG system
        chunks = [
            Chunk(content="text paragraph 1", from_url="http://law.com/fake_law/1"),
            Chunk(content="text paragraph 2", from_url="http://law.com/fake_law/2"),
        ]
        return chunks

    @goal(final=True)
    @worker(output_effects=["answer_in_law_domain"])
    def answer_law_question(self, user_input: str, law_chunks: List[Chunk]) -> str:
        # TODO: call LLM to synthesize the final answer
        return "The answer to the question is: xxx"

    @worker(output_effects=["finance_chunks"])
    def rag_in_finance_domain(self, user_input: str, routing_result: RoutingResult) -> List[Chunk]:
        if not routing_result.sucess or routing_result.domain != "finance":
            # TODO: return None or raise an Exception to stop here
            return None

        # TODO: query from RAG system
        chunks = [
            Chunk(content="text paragraph 1", from_url="http://finance.com/fake_url/1"),
            Chunk(content="text paragraph 2", from_url="http://finance.com/fake_url/2"),
        ]
        return chunks

    @goal(final=True)
    @worker(output_effects=["answer_in_finance_domain"])
    def answer_finance_question(self, user_input: str, finance_chunks: List[Chunk]) -> str:
        # TODO: call LLM to synthesize the final answer
        return "The answer to the question is: xxx"
    
@pytest.fixture
def chatbot():
    return RoutingExample_RAGChatbot()

@pytest.mark.asyncio
async def test_rag_chatbot(chatbot):
    # answer = await chatbot.process_async(
    #     user_input="How does monetary policy influence inflation and economic growth?"
    # )
    # assert answer == "The answer to the question is: xxx"
    pass
