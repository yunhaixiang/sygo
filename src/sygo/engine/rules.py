"""Ruleset definitions for engine, GTP, SGF, and training code."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Scoring(str, Enum):
    """Supported scoring families."""

    AREA = "area"


class KoRule(str, Enum):
    """Supported repetition policies."""

    SIMPLE = "simple"
    POSITIONAL_SUPERKO = "positional_superko"


@dataclass(frozen=True)
class Ruleset:
    """Concrete Go rules used by a game."""

    name: str
    scoring: Scoring
    suicide_allowed: bool
    ko_rule: KoRule
    komi_by_size: dict[int, float]
    play_out_disputes: bool

    def komi(self, board_size: int) -> float:
        """Return the configured komi for a board size."""

        return self.komi_by_size.get(board_size, self.komi_by_size[19])


CGOS_RULES = Ruleset(
    name="CGOS modified Tromp-Taylor",
    scoring=Scoring.AREA,
    suicide_allowed=False,
    ko_rule=KoRule.SIMPLE,
    komi_by_size={9: 7.0, 13: 7.5, 19: 7.5},
    play_out_disputes=True,
)

DEFAULT_RULES = CGOS_RULES
