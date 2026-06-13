from sygo.engine.board import Color, Point


def test_color_opponent() -> None:
    assert Color.BLACK.opponent is Color.WHITE
    assert Color.WHITE.opponent is Color.BLACK


def test_point_is_zero_indexed_value() -> None:
    assert Point(row=0, col=0).row == 0
