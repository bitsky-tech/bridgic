import pickle
from typing import Optional, Any, Dict

class MemoryCache:
    """
    In-memory, default rendering cache.
    """

    def __init__(self) -> None:
        self._cache: Dict[bytes, Any] = {}

    def get(self, key: Any) -> Optional[str]:
        return self._cache.get(pickle.dumps(key, pickle.HIGHEST_PROTOCOL))

    def set(self, key: Any, value: Any):
        self._cache[pickle.dumps(key, pickle.HIGHEST_PROTOCOL)] = value

    def clear(self) -> None:
        self._cache = {}