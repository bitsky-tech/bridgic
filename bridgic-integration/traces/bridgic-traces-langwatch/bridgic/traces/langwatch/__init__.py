from importlib.metadata import version
from ._langwatch_trace_callback import LangWatchTraceCallback
from ._utils import start_langwatch_trace

__version__ = version("bridgic-traces-langwatch")
__all__ = ["LangWatchTraceCallback", "start_langwatch_trace", "__version__"]
