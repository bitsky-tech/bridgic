from typing import Optional
from bridgic.core.automa.worker import WorkerCallbackBuilder
from langwatch.types import BaseAttributes
from ._langwatch_trace_callback import LangWatchTraceCallback

def start_langwatch_trace(
    api_key: Optional[str] = None,
    endpoint_url: Optional[str] = None,
    base_attributes: Optional[BaseAttributes] = None,
) -> None:
    """Start a LangWatch trace for a given project and service.

    Parameters
    ----------
    api_key : Optional[str], default=None
        The API key for the LangWatch tracing service, if none is provided, the `LANGWATCH_API_KEY` environment variable will be used.
    endpoint_url : Optional[str], default=None
        The URL of the LangWatch tracing service, if none is provided, the `LANGWATCH_ENDPOINT` environment variable will be used. If that is not provided, the default value will be `https://app.langwatch.ai`.
    base_attributes : Optional[BaseAttributes], default=None
        The base attributes to use for the LangWatch tracing client.
    
    Returns
    -------
    None
    """
    from bridgic.core.config import GlobalSetting
    builder = WorkerCallbackBuilder(
        LangWatchTraceCallback, 
        init_kwargs={"api_key": api_key, "endpoint_url": endpoint_url, "base_attributes": base_attributes}
    )
    GlobalSetting.add(callback_builder=builder)