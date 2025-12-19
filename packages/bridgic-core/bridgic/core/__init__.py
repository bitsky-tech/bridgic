# Core module initialization

from importlib.metadata import version

__version__ = version("bridgic-core")
__all__ = ["__version__"]