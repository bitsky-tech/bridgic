import pytest
from bridgic.core.automa import GoapAutoma, worker, goal
from typing import List
from pydantic import BaseModel

class Snippet(BaseModel):
    title: str
    content: str
    from_url: str

class SequentialExample_QuerySummarizer(GoapAutoma):

    @worker(output_effects=["text_in_english"])
    def translate_to_english(self, text_in_chinese: str) -> str:
        text_in_english = "A translator should be called on $text_in_chinese"
        return text_in_english

    @worker(output_effects=["keywords"])
    def extract_keywords(self, text_in_english: str) -> List[str]:
        keywords = ["translator", "should", "called"]
        return keywords

    @worker(output_effects=["query_results"])
    def query_by_keywords(self, keywords: List[str]) -> List[Snippet]:
        # TODO: query from database / web
        snippets = [
            Snippet(title="title1", content="content1", from_url="url1"),
            Snippet(title="title2", content="content2", from_url="url2"),
        ]
        return snippets

    @goal(final=True)
    @worker(output_effects=["summary"])
    def summarize_snippets(self, query_results: List[Snippet]) -> str:
        return "A summarizer should be called on $query_results"

@pytest.fixture
def query_summarizer():
    return SequentialExample_QuerySummarizer()

@pytest.mark.asyncio
async def test_query_summarizer(query_summarizer):
    pass
