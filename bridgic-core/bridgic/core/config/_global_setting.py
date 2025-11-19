"""
Global settings for the Bridgic framework.
"""

from typing import List, Optional, ClassVar, TYPE_CHECKING
from pydantic import BaseModel
from threading import Lock

if TYPE_CHECKING:
    from bridgic.core.automa.worker._worker_callback import WorkerCallbackBuilder


class GlobalSetting(BaseModel):
    """
    Global configuration settings for the Bridgic framework.

    This class implements a singleton pattern to provide centralized configuration
    that applies across all Automa instances. The main methods are:

    - `GlobalSetting.read()`: Get the singleton global setting instance.
    - `GlobalSetting.set()`: Set the specific fields of the global setting instance.

    Attributes
    ----------
    callback_builders : List[WorkerCallbackBuilder]
        Callback builders that will be automatically applied to all workers
        across all Automa instances.
    """
    model_config = {"arbitrary_types_allowed": True}

    callback_builders: List["WorkerCallbackBuilder"] = []
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
        callback_builders: Optional[List["WorkerCallbackBuilder"]] = None,
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
