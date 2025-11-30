import os
import sys
import subprocess
import time
import socket

from typing import Optional


def _wait_for_server(host: str, port: int, timeout: float = 10.0) -> bool:
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex((host, port))
            sock.close()
            if result == 0:
                time.sleep(0.5)
                return True
        except Exception:
            pass
        time.sleep(0.1)
    return False


class McpHttpServerProcess:
    """
    Context manager for managing MCP server processes.

    This class provides a convenient way to start an MCP server process,
    wait for it to be ready, and ensure it is properly cleaned up when
    the context exits, even if an exception occurs.

    Example
    -------
    >>> with McpHttpServerProcess(
    ...     server_script="path/to/server.py",
    ...     transport="streamable_http",
    ...     host="127.0.0.1",
    ...     port=8000,
    ... ) as server:
    ...     connection = create_connection(url=server.url)
    """

    def __init__(
        self,
        server_script: str,
        transport: str = "streamable_http",
        host: str = "127.0.0.1",
        port: int = 8000,
        startup_timeout: float = 10.0,
    ):
        self.server_script = os.path.abspath(server_script) if not os.path.isabs(server_script) else server_script
        self.transport = transport
        self.host = host
        self.port = port
        self.startup_timeout = startup_timeout
        self.process: Optional[subprocess.Popen] = None

    def __enter__(self) -> "McpHttpServerProcess":
        # Start the server process
        self.process = subprocess.Popen(
            [
                sys.executable, self.server_script,
                "--transport", self.transport,
                "--host", self.host,
                "--port", str(self.port),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait for the server to be ready
        if not _wait_for_server(self.host, self.port, timeout=self.startup_timeout):
            self._cleanup()
            raise RuntimeError(
                f"Server failed to start on {self.host}:{self.port} within {self.startup_timeout}s timeout"
            )
        
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cleanup()
        return False  # Don't suppress exceptions

    def _cleanup(self):
        if self.process is not None:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            finally:
                self.process = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/mcp"

