import asyncio
import threading
from typing import Optional, Any, ClassVar, Coroutine, TYPE_CHECKING
from weakref import WeakSet

from bridgic.core.types._error import McpServerConnectionError

if TYPE_CHECKING:
    from bridgic.core.mcp._mcp_server_connection import McpServerConnection


class McpServerConnectionManager:
    """
    Manages multiple MCP server connections, sharing a single thread and event loop.
    
    This manager ensures that all MCP operations run in a dedicated thread with its own
    event loop, avoiding issues with cross-thread event loop usage.
    """

    _default_instance_lock: ClassVar[threading.Lock] = threading.Lock()
    _default_instance: ClassVar["McpServerConnectionManager"] = None

    _name: str

    _connections_lock: threading.Lock
    _connections: WeakSet

    _thread: threading.Thread
    _loop: asyncio.AbstractEventLoop
    _shutdown: bool

    def __init__(self, name: str):
        self._name = name
        self._connections_lock = threading.Lock()
        self._connections = WeakSet()
        self._thread = None
        self._loop = None
        self._shutdown = False

    @classmethod
    def get_instance(cls) -> "McpServerConnectionManager":
        """
        Get the default manager instance.

        Returns
        -------
        McpServerConnectionManager
            The default manager instance.
        """
        if cls._default_instance is None:
            with cls._default_instance_lock:
                if cls._default_instance is None:
                    cls._default_instance = cls(name="default-mcp-manager")
        return cls._default_instance

    def register_connection(self, connection: "McpServerConnection"):
        """
        Register a connection into the manager.

        Parameters
        ----------
        connection : McpServerConnection
            The connection to register.
        """
        self._ensure_loop_running()
        with self._connections_lock:
            self._connections.add(connection)
        connection._manager = self

    def unregister_connection(self, connection: "McpServerConnection"):
        """
        Unregister a connection from the manager.
        
        Parameters
        ----------
        connection : McpServerConnection
            The connection to unregister.
        """
        with self._connections_lock:
            self._connections.discard(connection)

    def run_async(
        self,
        coro: Coroutine[Any, Any, Any],
        timeout: Optional[float] = None,
    ) -> Any:
        """
        Submit a coroutine to the manager's event loop and wait for the result.

        Parameters
        ----------
        coro : Coroutine
            The coroutine to run.
        timeout : Optional[float]
            Timeout in seconds. If None, no timeout.

        Returns
        -------
        Any
            The result of the coroutine execution.

        Raises
        ------
        RuntimeError
            If the event loop is not running.
        TimeoutError
            If the coroutine execution times out.
        """
        self._ensure_loop_running()

        if self._loop is None or not self._loop.is_running():
            raise RuntimeError(f"Event loop is not running in McpServerConnectionManager-[{self._name}]")

        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    def shutdown(self):
        """
        Shutdown the manager and stop the event loop.
        """
        with self._connections_lock:
            if self._loop is not None:
                # First, mark the manager is going to shutdown.
                self._shutdown = True
                # Then stop the event loop as soon as possible.
                self._loop.call_soon_threadsafe(self._loop.stop)
                if self._thread is not None:
                    self._thread.join(timeout=5)
                self._loop = None
                self._thread = None

    def _ensure_loop_running(self):
        """
        Ensure the manager's event loop is running in a dedicated thread.
        """
        def run_until_shutdown():
            asyncio.set_event_loop(self._loop)
            try:
                self._loop.run_forever()
            finally:
                self._loop.close()

        with self._connections_lock:
            if self._loop is not None and self._loop.is_running():
                return

            if self._shutdown:
                raise RuntimeError(f"McpServerConnectionManager-[{self._name}] has been shut down")

            self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(
                target=run_until_shutdown,
                daemon=True,
            )
            self._thread.start()
