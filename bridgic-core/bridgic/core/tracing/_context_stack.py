from __future__ import annotations

from contextvars import ContextVar
from typing import Generic, Optional, Tuple, TypeVar

T = TypeVar("T")


class ContextStack(Generic[T]):
    """
    Thin wrapper around ContextVar-based stacks.

    Useful for managing nested spans/traces where context propagation
    across async tasks is required.
    """

    def __init__(self, name: str):
        self._var: ContextVar[Tuple[T, ...]] = ContextVar(name, default=())

    def push(self, item: T) -> None:
        stack = self._var.get()
        self._var.set((*stack, item))

    def pop(self) -> Optional[T]:
        stack = self._var.get()
        if not stack:
            return None
        item = stack[-1]
        self._var.set(stack[:-1])
        return item

    def peek(self) -> Optional[T]:
        stack = self._var.get()
        return stack[-1] if stack else None

    def reset(self) -> None:
        self._var.set(())

