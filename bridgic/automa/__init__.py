# 这个包负责框架的编排层：
# bridgic.automa
# 它主要依赖bridgic.core包，不依赖任何框架外部概念。

from .automa import AutoMa

__all__ = ["AutoMa"]