"""
Global settings for the Bridgic framework.
"""

from typing import List, Optional, ClassVar
from pydantic import BaseModel
from threading import Lock

from bridgic.core.automa.worker._worker_callback import WorkerCallbackBuilder


class GlobalSetting(BaseModel):
    """
    Global settings for the Bridgic framework.
    
    This class uses a singleton pattern. The singleton instance is accessed via 
    `GlobalSetting.get()` and can be configured via `GlobalSetting.set()`.
    
    These settings apply to all Automa instances and are merged with Automa-level 
    and Worker-level settings. Global settings are not overridden by other levels; 
    instead, they are combined in the order: Global -> Automa -> Worker.
    
    Attributes
    ----------
    callback_builders : List[WorkerCallbackBuilder]
        Global callback builders that will be applied to all workers.
    """
    model_config = {"arbitrary_types_allowed": True}

    callback_builders: List[WorkerCallbackBuilder] = []
    """Global callback builders that will be applied to all workers."""

    # Singleton instance
    _instance: ClassVar[Optional["GlobalSetting"]] = None
    _lock: ClassVar[Lock] = Lock()

    @classmethod
    def read(cls) -> "GlobalSetting":
        """
        Get the singleton global setting instance.
        
        Returns
        -------
        GlobalSetting
            The singleton global setting instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def set(
        cls,
        callback_builders: Optional[List[WorkerCallbackBuilder]] = None,
    ) -> None:
        """
        Set global setting fields.
        
        This method allows you to update specific fields of the global setting
        without needing to create a complete GlobalSetting object.
        
        Parameters
        ----------
        callback_builders : Optional[List[WorkerCallbackBuilder]], optional
            Global callback builders that will be applied to all workers.
            If None, the current callback_builders are not changed.
        """
        instance = cls.read()
        with cls._lock:
            if callback_builders is not None:
                instance.callback_builders = callback_builders
