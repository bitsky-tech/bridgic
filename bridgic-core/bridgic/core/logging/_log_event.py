"""
Structured logging events for Bridgic Core.

This module provides structured event classes for logging various operations
within the Bridgic framework.
"""

import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Field


class EventType(str, Enum):
    """Types of events that can be logged.
    
    Attributes
    ----------
    WORKER_START : str
        Worker start event
    WORKER_END : str
        Worker end event
    WORKER_ERROR : str
        Worker error event
    WORKER_FERRY : str
        Worker ferry event
    AUTOMA_START : str
        Automa start event
    AUTOMA_END : str
        Automa end event
    AUTOMA_ERROR : str
        Automa error event
    MODEL_CALL : str
        Model call event
    CACHE_HIT : str
        Cache hit event
    CACHE_MISS : str
        Cache miss event
    CACHE_SET : str
        Cache set event
    CACHE_DELETE : str
        Cache delete event
    """
    WORKER_START = "WorkerStart"
    WORKER_END = "WorkerEnd"
    WORKER_ERROR = "WorkerError"
    WORKER_FERRY = "WorkerFerry"
    AUTOMA_START = "AutomaStart"
    AUTOMA_END = "AutomaEnd"
    AUTOMA_ERROR = "AutomaError"
    MODEL_CALL = "ModelCall"
    CACHE_HIT = "CacheHit"
    CACHE_MISS = "CacheMiss"
    CACHE_SET = "CacheSet"
    CACHE_DELETE = "CacheDelete"


class BaseEvent(BaseModel):
    """Base class for all structured events.
    
    Attributes
    ----------
    event_id : str
        Unique identifier for this event
    timestamp : datetime
        Timestamp when the event occurred
    event_type : EventType
        Type of the event
    source : str, optional
        Source component that generated the event
    metadata : Dict[str, Any]
        Additional metadata about the event
    """
    
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    """Unique identifier for this event."""
    
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    """Timestamp when the event occurred."""
    
    event_type: EventType
    """Type of the event."""
    
    source: Optional[str] = None
    """Source component that generated the event."""
    
    metadata: Dict[str, Any] = Field(default_factory=dict)
    """Additional metadata about the event."""
    
    def to_json(self) -> str:
        """Convert the event to JSON string."""
        return json.dumps(self.model_dump(), default=str)
    
    def __str__(self) -> str:
        """String representation of the event."""
        return self.to_json()


class WorkerEvent(BaseEvent):
    """Base class for worker-related events.
    
    Attributes
    ----------
    worker_id : str
        Identifier of the worker
    worker_name : str
        Name of the worker
    worker_key : str, optional
        Key of the worker in the automa graph (for graph visualization)
    execution_id : str, optional
        Identifier for the execution context
    """
    
    worker_id: str
    """Identifier of the worker."""
    
    worker_name: str
    """Name of the worker."""
    
    worker_key: Optional[str] = None
    """Key of the worker in the automa graph (for graph visualization)."""
    
    execution_id: Optional[str] = None
    """Identifier for the execution context."""


class WorkerStartEvent(WorkerEvent):
    """Event emitted when a worker starts execution.
    
    Attributes
    ----------
    input_data : Dict[str, Any]
        Input data passed to the worker
    dependencies : List[str], optional
        List of worker keys this worker depends on (for graph visualization)
    triggered_by : str, optional
        Worker key that triggered this execution (predecessor in execution)
    """
    
    event_type: EventType = EventType.WORKER_START
    
    input_data: Dict[str, Any] = Field(default_factory=dict)
    """Input data passed to the worker."""
    
    dependencies: Optional[List[str]] = None
    """List of worker keys this worker depends on (for graph visualization)."""
    
    triggered_by: Optional[str] = None
    """Worker key that triggered this execution (predecessor in execution)."""
    
    def __str__(self) -> str:
        return json.dumps({
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "worker_id": self.worker_id,
            "worker_name": self.worker_name,
            "worker_key": self.worker_key,
            "execution_id": self.execution_id,
            "input_data": self.input_data,
            "dependencies": self.dependencies,
            "triggered_by": self.triggered_by,
            "source": self.source,
            "metadata": self.metadata,
        })


class WorkerEndEvent(WorkerEvent):
    """Event emitted when a worker completes execution.
    
    Attributes
    ----------
    output_data : Dict[str, Any]
        Output data produced by the worker
    execution_time : float, optional
        Time taken to execute the worker in seconds
    """
    
    event_type: EventType = EventType.WORKER_END
    
    output_data: Dict[str, Any] = Field(default_factory=dict)
    """Output data produced by the worker."""
    
    execution_time: Optional[float] = None
    """Time taken to execute the worker in seconds."""
    
    def __str__(self) -> str:
        return json.dumps({
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "worker_id": self.worker_id,
            "worker_name": self.worker_name,
            "worker_key": self.worker_key,
            "execution_id": self.execution_id,
            "output_data": self.output_data,
            "execution_time": self.execution_time,
            "source": self.source,
            "metadata": self.metadata,
        })


class WorkerErrorEvent(WorkerEvent):
    """Event emitted when a worker encounters an error.
    
    Attributes
    ----------
    error_type : str
        Type of the error
    error_message : str
        Error message
    error_traceback : str, optional
        Error traceback if available
    execution_time : float, optional
        Time taken before the error occurred in seconds
    """
    
    event_type: EventType = EventType.WORKER_ERROR
    
    error_type: str
    """Type of the error."""
    
    error_message: str
    """Error message."""
    
    error_traceback: Optional[str] = None
    """Error traceback if available."""
    
    execution_time: Optional[float] = None
    """Time taken before the error occurred in seconds."""
    
    def __str__(self) -> str:
        return json.dumps({
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "worker_id": self.worker_id,
            "worker_name": self.worker_name,
            "worker_key": self.worker_key,
            "execution_id": self.execution_id,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "error_traceback": self.error_traceback,
            "execution_time": self.execution_time,
            "source": self.source,
            "metadata": self.metadata,
        })


class WorkerFerryEvent(WorkerEvent):
    """Event emitted when a worker performs a ferry_to operation.
    
    Attributes
    ----------
    from_worker_key : str
        Key of the worker initiating the ferry
    to_worker_key : str
        Key of the target worker
    ferry_data : Dict[str, Any]
        Data being passed in the ferry operation
    automa_name : str, optional
        Name of the automa containing these workers
    automa_id : str, optional
        ID of the automa containing these workers
    """
    
    event_type: EventType = EventType.WORKER_FERRY
    """Type of the event."""
    
    from_worker_key: str
    """Key of the worker initiating the ferry."""
    
    to_worker_key: str
    """Key of the target worker."""
    
    ferry_data: Dict[str, Any] = Field(default_factory=dict)
    """Data being passed in the ferry operation."""
    
    automa_name: Optional[str] = None
    """Name of the automa containing these workers."""
    
    automa_id: Optional[str] = None
    """ID of the automa containing these workers."""
    
    def __str__(self) -> str:
        return json.dumps({
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "worker_id": self.worker_id,
            "worker_name": self.worker_name,
            "worker_key": self.worker_key,
            "from_worker_key": self.from_worker_key,
            "to_worker_key": self.to_worker_key,
            "ferry_data": self.ferry_data,
            "automa_name": self.automa_name,
            "automa_id": self.automa_id,
            "execution_id": self.execution_id,
            "source": self.source,
            "metadata": self.metadata,
        })


class AutomaEvent(BaseEvent):
    """Base class for automa-related events.
    
    Attributes
    ----------
    automa_id : str
        Identifier of the automa
    automa_name : str
        Name of the automa
    execution_id : str, optional
        Identifier for the execution context
    """
    
    automa_id: str
    """Identifier of the automa."""
    
    automa_name: str
    """Name of the automa."""
    
    execution_id: Optional[str] = None
    """Identifier for the execution context."""


class AutomaStartEvent(AutomaEvent):
    """Event emitted when an automa starts execution.
    
    Attributes
    ----------
    workflow_data : Dict[str, Any]
        Workflow data for the automa
    """
    
    event_type: EventType = EventType.AUTOMA_START
    
    workflow_data: Dict[str, Any] = Field(default_factory=dict)
    """Workflow data for the automa."""
    
    def __str__(self) -> str:
        return json.dumps({
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "automa_id": self.automa_id,
            "automa_name": self.automa_name,
            "execution_id": self.execution_id,
            "workflow_data": self.workflow_data,
            "source": self.source,
            "metadata": self.metadata,
        })


class AutomaEndEvent(AutomaEvent):
    """Event emitted when an automa completes execution.
    
    Attributes
    ----------
    result_data : Dict[str, Any]
        Result data produced by the automa
    execution_time : float, optional
        Time taken to execute the automa in seconds
    """
    
    event_type: EventType = EventType.AUTOMA_END
    
    result_data: Dict[str, Any] = Field(default_factory=dict)
    """Result data produced by the automa."""
    
    execution_time: Optional[float] = None
    """Time taken to execute the automa in seconds."""
    
    def __str__(self) -> str:
        return json.dumps({
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "automa_id": self.automa_id,
            "automa_name": self.automa_name,
            "execution_id": self.execution_id,
            "result_data": self.result_data,
            "execution_time": self.execution_time,
            "source": self.source,
            "metadata": self.metadata,
        })


class AutomaErrorEvent(AutomaEvent):
    """Event emitted when an automa encounters an error.
    
    Attributes
    ----------
    error_type : str
        Type of the error
    error_message : str
        Error message
    error_traceback : str, optional
        Error traceback if available
    execution_time : float, optional
        Time taken before the error occurred in seconds
    """
    
    event_type: EventType = EventType.AUTOMA_ERROR
    
    error_type: str
    """Type of the error."""
    
    error_message: str
    """Error message."""
    
    error_traceback: Optional[str] = None
    """Error traceback if available."""
    
    execution_time: Optional[float] = None
    """Time taken before the error occurred in seconds."""
    
    def __str__(self) -> str:
        return json.dumps({
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "automa_id": self.automa_id,
            "automa_name": self.automa_name,
            "execution_id": self.execution_id,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "error_traceback": self.error_traceback,
            "execution_time": self.execution_time,
            "source": self.source,
            "metadata": self.metadata,
        })


class ModelCallEvent(BaseEvent):
    """Event emitted when a model is called.
    
    Attributes
    ----------
    model_name : str
        Name of the model being called
    prompt_tokens : int
        Number of tokens in the prompt
    completion_tokens : int
        Number of tokens in the completion
    total_tokens : int
        Total number of tokens used
    model_response : Dict[str, Any]
        Response from the model
    execution_time : float, optional
        Time taken for the model call in seconds
    """
    
    event_type: EventType = EventType.MODEL_CALL
    
    model_name: str
    """Name of the model being called."""
    
    prompt_tokens: int
    """Number of tokens in the prompt."""
    
    completion_tokens: int
    """Number of tokens in the completion."""
    
    total_tokens: int
    """Total number of tokens used."""
    
    model_response: Dict[str, Any] = Field(default_factory=dict)
    """Response from the model."""
    
    execution_time: Optional[float] = None
    """Time taken for the model call in seconds."""
    
    def __str__(self) -> str:
        return json.dumps({
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "model_name": self.model_name,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "model_response": self.model_response,
            "execution_time": self.execution_time,
            "source": self.source,
            "metadata": self.metadata,
        })




class CacheEvent(BaseEvent):
    """Base class for cache-related events.
    
    Attributes
    ----------
    cache_key : str
        Cache key
    cache_operation : str
        Cache operation (get, set, delete)
    """
    
    cache_key: str
    """Cache key."""
    
    cache_operation: str
    """Cache operation (get, set, delete)."""


class CacheHitEvent(CacheEvent):
    """Event emitted when a cache hit occurs."""
    
    event_type: EventType = EventType.CACHE_HIT
    
    cache_operation: str = "get"
    
    def __str__(self) -> str:
        return json.dumps({
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "cache_key": self.cache_key,
            "cache_operation": self.cache_operation,
            "source": self.source,
            "metadata": self.metadata,
        })


class CacheMissEvent(CacheEvent):
    """Event emitted when a cache miss occurs."""
    
    event_type: EventType = EventType.CACHE_MISS
    
    cache_operation: str = "get"
    
    def __str__(self) -> str:
        return json.dumps({
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "cache_key": self.cache_key,
            "cache_operation": self.cache_operation,
            "source": self.source,
            "metadata": self.metadata,
        })


class CacheSetEvent(CacheEvent):
    """Event emitted when a cache set operation occurs."""
    
    event_type: EventType = EventType.CACHE_SET
    
    cache_operation: str = "set"
    
    def __str__(self) -> str:
        return json.dumps({
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "cache_key": self.cache_key,
            "cache_operation": self.cache_operation,
            "source": self.source,
            "metadata": self.metadata,
        })


class CacheDeleteEvent(CacheEvent):
    """Event emitted when a cache delete operation occurs."""
    
    event_type: EventType = EventType.CACHE_DELETE
    
    cache_operation: str = "delete"
    
    def __str__(self) -> str:
        return json.dumps({
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "cache_key": self.cache_key,
            "cache_operation": self.cache_operation,
            "source": self.source,
            "metadata": self.metadata,
        })


# Union type for all events
BridgicEvent = Union[
    WorkerStartEvent,
    WorkerEndEvent,
    WorkerErrorEvent,
    WorkerFerryEvent,
    AutomaStartEvent,
    AutomaEndEvent,
    AutomaErrorEvent,
    ModelCallEvent,
    CacheHitEvent,
    CacheMissEvent,
    CacheSetEvent,
    CacheDeleteEvent,
]
