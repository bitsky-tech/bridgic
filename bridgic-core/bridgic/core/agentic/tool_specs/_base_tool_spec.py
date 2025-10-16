from typing import Optional, Union, Dict, Any
from abc import abstractmethod
from bridgic.core.types._serialization import Serializable
from bridgic.core.model.types import Tool
from bridgic.core.automa.worker import Worker
from typing_extensions import override

class ToolSpec(Serializable):
    """
    ToolSpec is an abstract class that represents a tool specification that describes all necessary information about a tool used by the LLM. 

    ToolSpec and its subclasses are responsible for providing four categories of interfaces:
    1. Transformations to LLM Tool: `to_tool`.
    2. Worker Creation: `create_worker`.
    3. Serialization and Deserialization.
    4. ToolSpec initialization from raw resources: `from_raw`.
    """
    _tool_id: Optional[Union[str, int]]
    """The unique ID of the tool, used to uniquely identify a tool across the entire system. This tool can be of various types."""
    _tool_name: Optional[str]
    """The name of the tool to be called"""
    _tool_description: Optional[str]
    """A description of what the tool does, used by the model to choose when and how to call the tool."""
    _tool_parameters: Optional[Dict[str, Any]]
    """The JSON schema of the tool's parameters"""

    def __init__(
        self,
        tool_name: Optional[str] = None,
        tool_description: Optional[str] = None,
        tool_parameters: Optional[Dict[str, Any]] = None,
    ):
        self._tool_id = None
        self._tool_name = tool_name
        self._tool_description = tool_description
        self._tool_parameters = tool_parameters

    @property
    def tool_name(self) -> Optional[str]:
        return self._tool_name

    @property
    def tool_description(self) -> Optional[str]:
        return self._tool_description

    @property
    def tool_parameters(self) -> Optional[Dict[str, Any]]:
        return self._tool_parameters

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(tool_name={self._tool_name}, tool_description={self._tool_description}, tool_parameters={self._tool_parameters})"
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(tool_name={self._tool_name}, tool_description={self._tool_description}, tool_parameters={self._tool_parameters})>"

    ###############################################################
    ######## Part One of interfaces: Transformations to Tool ######
    ###############################################################

    @abstractmethod
    def to_tool(self) -> Tool:
        """
        Transform this ToolSpec to a `Tool` object used by LLM.

        Returns
        -------
        Tool
            A `Tool` object that can be used by LLM.
        """
        ...

    ###############################################################
    ######## Part Two of interfaces: Worker Creation ##############
    ###############################################################

    @abstractmethod
    def create_worker(self) -> Worker:
        """
        Create a Worker from the information included in this ToolSpec.

        Returns
        -------
        Worker
            A new `Worker` object that can be added to an Automa to execute the tool.
        """
        ...

    ###############################################################
    ######## Part Three of interfaces: 
    ######## Serialization and Deserialization ####################
    ###############################################################

    @override
    def dump_to_dict(self) -> Dict[str, Any]:
        state_dict = {}
        if self._tool_id:
            state_dict["tool_id"] = self._tool_id
        if self._tool_name:
            state_dict["tool_name"] = self._tool_name
        if self._tool_description:
            state_dict["tool_description"] = self._tool_description
        if self._tool_parameters:
            state_dict["tool_parameters"] = self._tool_parameters
        return state_dict

    @override
    def load_from_dict(self, state_dict: Dict[str, Any]) -> None:
        self._tool_id = state_dict.get("tool_id")
        self._tool_name = state_dict.get("tool_name")
        self._tool_description = state_dict.get("tool_description")
        self._tool_parameters = state_dict.get("tool_parameters")

    ###############################################################
    ######## Part Four of interfaces: 
    ######## ToolSpec initialization from raw resources ###########
    ######## `from(...)`: See subclasses for details ##############
    ###############################################################
