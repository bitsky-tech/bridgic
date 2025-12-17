from typing import Optional
from bridgic.core.automa.worker import WorkerCallbackBuilder
from ._opik_trace_callback import OpikTraceCallback

def start_opik_trace(
    project_name: Optional[str] = None,
    workspace: Optional[str] = None,
    host: Optional[str] = None,
    api_key: Optional[str] = None,
    use_local: bool = False,
) -> None:
    """Start a Opik trace for a given project and service.

    Parameters
    ----------
    project_name : Optional[str], default=None
        The name of the project. If None, uses `Default Project` project name.
    workspace : Optional[str], default=None
        The name of the workspace. If None, uses `default` workspace name.
    host : Optional[str], default=None
        The host URL for the Opik server. If None, it will default to `https://www.comet.com/opik/api`.
    api_key : Optional[str], default=None
        The API key for Opik. This parameter is ignored for local installations.
    use_local : bool, default=False
        Whether to use local Opik server.

    Returns
    -------
    None
    """
    from bridgic.core.config import GlobalSetting
    builder = WorkerCallbackBuilder(
        OpikTraceCallback, 
        init_kwargs={"project_name": project_name, "workspace": workspace, "host": host, "api_key": api_key, "use_local": use_local}
    )
    GlobalSetting.add(callback_builder=builder)