"""Core Go board primitives.

This module is intentionally small for the initial scaffold. It will become the
rules authority used by GTP, SGF replay, self-play, and the GUI bridge.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Color(str, Enum):
    """Stone colors used by the engine."""

    BLACK = "black"
    WHITE = "white"

    @property
    def opponent(self) -> "Color":
        return Color.WHITE if self is Color.BLACK else Color.BLACK


@dataclass(frozen=True, slots=True)
class Point:
    """A zero-indexed board point."""

    row: int
    col: int


@dataclass(slots=True)
class Move:
    """A Go move. `point=None` represents pass."""

    color: Color
    point: Point | None
