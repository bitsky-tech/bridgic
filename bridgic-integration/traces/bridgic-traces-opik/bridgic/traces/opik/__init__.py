from importlib.metadata import version
from ._opik_trace_callback import OpikTraceCallback
from ._utils import start_opik_trace

__version__ = version("bridgic-traces-opik")
__all__ = ["OpikTraceCallback", "start_opik_trace", "__version__"]
