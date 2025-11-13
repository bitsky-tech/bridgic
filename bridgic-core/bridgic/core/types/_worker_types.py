from typing import Dict, Any, Optional, List
from bridgic.core.automa.args import ArgsMappingRule
from pydantic import BaseModel


class WorkerExecutionContext(BaseModel):
    """
    Execution context information for worker logging and tracing.
    
    Contains information about the worker's execution environment including
    parent Automa, nesting level, key, dependencies, etc.
    """
    key: Optional[str] = None
    dependencies: Optional[List[str]] = None
    parent_automa_name: Optional[str] = None
    parent_automa_class: Optional[str] = None
    nesting_level: int = 0
    top_automa_name: Optional[str] = None
    is_output: bool = False
    is_start: bool = False
    local_space: Optional[Dict[str, Any]] = None
    args_mapping_rule: Optional[ArgsMappingRule] = None

    def to_metadata_dict(self) -> dict:
        """
        Convert to dictionary excluding key and dependencies.
        Used for metadata that doesn't need these fields separately.
        """
        return self.model_dump(exclude={"key", "dependencies"})
    
    def to_dict(self) -> dict:
        """
        Convert to full dictionary including all fields.
        """
        return self.model_dump()

