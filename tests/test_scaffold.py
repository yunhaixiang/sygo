from sygo.engine.board import Color, Point
from sygo.engine.rules import DEFAULT_RULES


def test_color_opponent() -> None:
    assert Color.BLACK.opponent is Color.WHITE
    assert Color.WHITE.opponent is Color.BLACK


def test_point_is_zero_indexed_value() -> None:
    assert Point(row=0, col=0).row == 0


def test_default_rules_are_cgos_target() -> None:
    assert DEFAULT_RULES.name == "CGOS modified Tromp-Taylor"
    assert DEFAULT_RULES.komi(9) == 7.0
