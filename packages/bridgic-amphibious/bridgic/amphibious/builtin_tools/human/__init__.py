"""Human-in-the-loop built-in tools.

Nothing is auto-injected. Declare ``request_human_tool`` on the OTA context
used by a run when an LLM-driven ``ThinkUnit`` should be able to ask a human::

    from bridgic.amphibious import OTAContext, request_human_tool

    class MyOTAContext(OTAContext):
        pass

    MyOTAContext.tool(request_human_tool)

Deterministic workflow and hook code can instead yield ``HumanCall``.
"""

from .request_human import request_human_tool

__all__ = ["request_human_tool"]
