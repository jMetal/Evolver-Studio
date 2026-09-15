"""Minimal Ok/Err result type for I/O-boundary code (see CODING_GUIDELINES.md)."""

from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(slots=True, frozen=True)
class Ok(Generic[T]):
    """A successful result carrying a value."""

    value: T


@dataclass(slots=True, frozen=True)
class Err:
    """A failed result carrying a human-readable message."""

    message: str
