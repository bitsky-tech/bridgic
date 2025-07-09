import pytest
from bridgic.automa import LLMPlanningAutoma, descriptive_worker, PlanningStrategy
from bridgic.core import LLM
from pydantic import BaseModel
from typing import List
import random
from bridgic.types.common_types import LLMOutputFormat

class File(BaseModel):
    file_path: str
    is_dir: bool

class ReActExample_IntelligentFileBrowser(LLMPlanningAutoma):
    def __init__(self):
        descriptive_goal = "Browse as many as possible files in the file system, and read the content of each file to find out the most relevant information to {person_name}"
        super().__init__(
            descriptive_goal=descriptive_goal,
            planning_llm=LLM(), # TODO: upgrade to a specific LLM model implementation
            expected_output_format=LLMOutputFormat.Json,
        )

    @descriptive_worker()
    def browse_files(self, start_dir_path: str) -> List[File]:
        """
        Browse the directory specified by the parameter `start_dir_path`, and return all files and directories under that directory.
        """
        # TODO: query the file system
        file_or_dir_list = [
            File(file_path="/home/user/documents/biography", is_dir=True),
            File(file_path="/home/user/documents/news.txt", is_dir=False),
            File(file_path="/home/user/documents/Musk.txt", is_dir=False),
        ]
        return file_or_dir_list

    @descriptive_worker(canonical_description="Read the content of the file specified by the parameter `file_path`")
    def read_file(self, file_path: str) -> str:
        # TODO: call the file system to read the file content
        contents = [
            "Elon Musk is a billionaire entrepreneur and the CEO of Tesla, SpaceX, and Neuralink.",
            "The latest climate report shows increasing global temperatures and rising sea levels.",
            "SpaceX successfully launched another batch of Starlink satellites into orbit under Musk's leadership.", 
            "A new species of deep sea fish was discovered off the coast of New Zealand.",
            "Elon Musk announced Tesla's plans for a new affordable electric vehicle model.",
            "Scientists published breakthrough research on quantum computing algorithms."
        ]
        content = random.choice(contents)
        return content

@pytest.fixture
def my_browser():
    return ReActExample_IntelligentFileBrowser()

@pytest.mark.asyncio
async def test_my_browser(my_browser):
    # result = await my_browser.process_async(
    #     automa_context={
    #         "person_name": "Elon Musk"
    #     }
    # )
    pass




