import asyncio
import threading
import weakref
from typing import Optional, Any, ClassVar, Coroutine, Awaitable, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from bridgic.protocols.mcp._mcp_server_connection import McpServerConnection

from bridgic.protocols.mcp._error import McpServerConnectionError


class McpServerConnectionManager:
    """
    Manages multiple MCP server connections, sharing a single thread and event loop.
    
    This manager ensures that all MCP operations run in a dedicated thread with its own
    event loop, avoiding issues with cross-thread event loop usage.
    """

    _instances_lock: ClassVar[threading.Lock] = threading.Lock()
    _instances: ClassVar[Dict[str, "McpServerConnectionManager"]] = {}
    _connection_to_manager: ClassVar[Dict[str, str]] = {}

    _name: str

    _connections_lock: threading.Lock
    _connections: weakref.WeakValueDictionary[str, "McpServerConnection"]

    _thread: threading.Thread
    _loop: asyncio.AbstractEventLoop
    _shutdown: bool

    def __init__(self, name: str):
        self._name = name
        self._connections_lock = threading.Lock()
        self._connections = weakref.WeakValueDictionary()
        self._thread = None
        self._loop = None
        self._shutdown = False

        cls = type(self)

        # Automatically register this instance in the class-level registry.
        with cls._instances_lock:
            if name in cls._instances:
                raise ValueError(f"Manager with name '{name}' already exists")
            cls._instances[name] = self

    @classmethod
    def get_instance(cls, manager_name: str = "default-mcp-manager") -> "McpServerConnectionManager":
        """
        Get a manager instance by name, creating it if it doesn't exist.

        Parameters
        ----------
        manager_name : str
            The name of the manager instance to retrieve. Defaults to "default-mcp-manager".

        Returns
        -------
        McpServerConnectionManager
            The manager instance with the specified name.
        """
        if manager_name not in cls._instances:
            return cls(name=manager_name)
        return cls._instances[manager_name]

    @classmethod
    def get_connection(cls, connection_name: str) -> Optional["McpServerConnection"]:
        """
        Get a connection by its name across all manager instances.

        It first finds the manager that owns the connection using the class-level 
        connection-to-manager mapping, then retrieves the connection from that manager.

        Parameters
        ----------
        connection_name : str
            The name of the connection to retrieve.

        Returns
        -------
        Optional[McpServerConnection]
            The connection with the specified name, or None if not found, the manager
            doesn't exist, or the connection has been garbage collected.

        Raises
        ------
        KeyError
            If the connection is not found.
        """
        manager_name = cls._connection_to_manager.get(connection_name)
        manager = cls.get_instance(manager_name) if manager_name is not None else None

        if manager is None:
            raise KeyError(
                f"McpServerConnectionManager-[{manager_name}] is not found: "
                f"connection_name={connection_name}"
            )

        return manager.get_connection_by_name(connection_name)

    def register_connection(self, connection: "McpServerConnection"):
        """
        Register a connection into the manager.

        Parameters
        ----------
        connection : McpServerConnection
            The connection to register.

        Raises
        ------
        McpServerConnectionError
            If a connection with the same name is already registered.
        """
        cls = type(self)
        if connection.name in cls._connection_to_manager:
            raise McpServerConnectionError(
                f"A connection with the name '{connection.name}' is already registered."
            )

        with cls._instances_lock:
            cls._connection_to_manager[connection.name] = self._name

        with self._connections_lock:
            self._connections[connection.name] = connection

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
            self._connections.pop(connection.name, None)

        cls = type(self)
        with cls._instances_lock:
            cls._connection_to_manager.pop(connection.name, None)

        connection._manager = None

    def get_connection_by_name(self, name: str) -> Optional["McpServerConnection"]:
        """
        Get a connection by its name from this manager instance.

        This method looks up a registered connection by its name within this manager.
        If the connection has been garbage collected, `weakref.WeakValueDictionary` will
        automatically remove it and None will be returned.

        Parameters
        ----------
        name : str
            The name of the connection to retrieve.

        Returns
        -------
        Optional[McpServerConnection]
            The connection with the specified name, or None if not found or has been
            garbage collected.

        Raises
        ------
        KeyError
            If the connection is not found.
        """
        if name not in self._connections:
            raise KeyError(
                f"Connection '{name}' is not found in McpServerConnectionManager-[{self._name}]"
            )

        return self._connections.get(name)

    def run_sync(
        self,
        coro: Coroutine[Any, Any, Any],
        timeout: Optional[float] = None,
    ) -> Any:
        """
        Submit a coroutine to the manager's event loop and wait for the result synchronously.

        This method blocks until the coroutine completes, suitable for use in synchronous contexts.

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

    async def run_async(
        self,
        coro: Coroutine[Any, Any, Any],
        timeout: Optional[float] = None,
    ) -> Any:
        """
        Submit a coroutine to the manager's event loop and await its result in a non-blocking way.

        This method submits the coroutine to the manager's dedicated event loop, and then 
        waits for its completion in a non-blocking way.

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
        asyncio_future = asyncio.wrap_future(future)

        if timeout is not None:
            return await asyncio.wait_for(asyncio_future, timeout=timeout)
        else:
            return await asyncio_future

    def shutdown(self):
        """
        Shutdown the manager and stop the event loop.

        This method also removes the manager from the class-level registry and cleans up
        all connection-to-manager mappings for connections registered with this manager.
        """
        # Mark the manager is going to shutdown.
        self._shutdown = True

        # Stop the event loop as soon as possible.
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)

        # Join the thread to wait for it to finish.
        if self._thread is not None:
            self._thread.join(timeout=5)

        self._loop = None
        self._thread = None

        cls = type(self)

        # Remove the registered item.
        with cls._instances_lock:
            # Remove the manager from the class-level registry.
            cls._instances.pop(self._name, None)

        # Clean up connection-to-manager mappings.
        conn_names_to_remove = [
            conn_name for conn_name, mgr_name in cls._connection_to_manager.items() if mgr_name == self._name
        ]
        with cls._instances_lock:
            for conn_name in conn_names_to_remove:
                cls._connection_to_manager.pop(conn_name, None)

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

