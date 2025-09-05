from bridgic.core.automa import GoapAutoma, worker, goal
from typing import List

class ParallelizationExample_MultipleLLMCall(GoapAutoma):
    @worker(output_effects=["abstract, main_text, conclution"])
    def get_writing_sections(self, writing_id: str) -> List[str]:
        # TODO: Get different parts of a writing content through writing_id
        abstract = "xxx..."
        main_text = "xxx..."
        conclution = "xxx..."
        # This example is different from the dynamic workers scenario in that: the number of subtasks is known in advance
        # Note: The length of the returned list here should match the length of output_effects specified in @worker
        return [abstract, main_text, conclution]
    
    @worker(output_effects=["abstract_review"])
    def review_abstract(self, abstract: str) -> str:
        # TODO: Call LLM to review the abstract
        return "abstract review xxx"
    
    @worker(output_effects=["main_text_review"])
    def review_main_text(self, main_text: str) -> str:
        # TODO: Call LLM to review the main text
        return "main text review xxx"
    
    @worker(output_effects=["conclution_review"])
    def review_conclution(self, conclution: str) -> str:
        # TODO: Call LLM to review the conclution
        return "conclution review xxx"
    
    @goal(final=True)
    @worker(output_effects=["final_review"])
    def synthesize_reviews(
        self, 
        abstract_review: str,
        main_text_review: str,
        conclution_review: str,
        ) -> str:
        # TODO: Call LLM to synthesize the reviews
        return "final review xxx"