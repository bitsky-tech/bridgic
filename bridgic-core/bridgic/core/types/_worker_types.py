from typing import Optional, List
from pydantic import BaseModel


class WorkerExecutionContext(BaseModel):
    """
    Execution context information for worker logging and tracing.
    
    Contains information about the worker's execution environment including
    parent Automa, nesting level, worker_key, dependencies, etc.
    """
    worker_key: Optional[str] = None
    dependencies: Optional[List[str]] = None
    parent_automa_name: Optional[str] = None
    parent_automa_id: Optional[str] = None
    parent_automa_class: Optional[str] = None
    nesting_level: int = 0
    top_automa_name: Optional[str] = None
    top_automa_id: Optional[str] = None

    def to_metadata_dict(self) -> dict:
        """
        Convert to dictionary excluding worker_key and dependencies.
        Used for metadata that doesn't need these fields separately.
        """
        return self.model_dump(exclude={"worker_key", "dependencies"})
    
    def to_dict(self) -> dict:
        """
        Convert to full dictionary including all fields.
        """
        return self.model_dump()

