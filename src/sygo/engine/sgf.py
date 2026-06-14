"""Smart Game Format integration point."""

from __future__ import annotations

from dataclasses import dataclass, field

from sygo.engine.board import Move


@dataclass
class GameRecord:
    """A parsed or generated Go game record."""

    board_size: int = 19
    moves: list[Move] = field(default_factory=list)
    komi: float = 7.5


def parse_sgf(_content: str) -> GameRecord:
    """Parse SGF content.

    Placeholder for a standards-compliant parser.
    """

    raise NotImplementedError("SGF parsing is not implemented yet.")


def write_sgf(_record: GameRecord) -> str:
    """Serialize a game record as SGF."""

    raise NotImplementedError("SGF writing is not implemented yet.")
