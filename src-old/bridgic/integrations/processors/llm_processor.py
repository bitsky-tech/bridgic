from bridgic.automa.module import Module
from bridgic.automa.bridge.data_model import ModuleDataType


# 临时代码，后续需要为LLM
class LLMClient:
    def __init__(self):
        self.llm = llm
    



class LLM(Module):

    def __init__(self, llm: LLMClient):
        self.llm = llm


    def acall(self, in_data: ModuleDataType) -> ModuleDataType:
        return self.llm.acall(in_data)