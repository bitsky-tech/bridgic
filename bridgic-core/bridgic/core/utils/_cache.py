import pickle
from typing import Optional, Any, Dict
from bridgic.core.logging import get_event_logger
from bridgic.core.constants import DEFAULT_CACHE_SOURCE_PREFIX

class MemoryCache:
    """
    In-memory, default rendering cache.
    """

    def __init__(self) -> None:
        self._cache: Dict[bytes, Any] = {}
        # Initialize logger
        self._logger = get_event_logger(f"{DEFAULT_CACHE_SOURCE_PREFIX}Component")

    def get(self, key: Any) -> Optional[str]:
        cache_key = str(key)
        result = self._cache.get(pickle.dumps(key, pickle.HIGHEST_PROTOCOL))
        
        if result is not None:
            # Log cache hit
            self._logger.log_cache_hit(
                cache_key=cache_key,
                metadata={"cache_size": len(self._cache), "operation": "get"}
            )
        else:
            # Log cache miss
            self._logger.log_cache_miss(
                cache_key=cache_key,
                metadata={"cache_size": len(self._cache), "operation": "get"}
            )
        
        return result

    def set(self, key: Any, value: Any):
        cache_key = str(key)
        self._cache[pickle.dumps(key, pickle.HIGHEST_PROTOCOL)] = value
        
        # Log cache set operation
        self._logger.log_cache_set(
            cache_key=cache_key,
            metadata={"cache_size": len(self._cache)}
        )

    def clear(self) -> None:
        old_size = len(self._cache)
        self._cache = {}
        
        # Log cache clear operation
        self._logger.log_cache_delete(
            cache_key="*",
            metadata={"old_cache_size": old_size, "operation": "clear"}
        )