import pytest

from sygo.engine.board import EMPTY, WHITE, Color, GameState, IllegalMoveError, Point


def test_capture_removes_surrounded_group() -> None:
    state = GameState(size=3)
    state = state.play(Point(1, 1))
    state = state.play(Point(0, 1))
    state = state.play(None)
    state = state.play(Point(1, 0))
    state = state.play(None)
    state = state.play(Point(1, 2))
    state = state.play(None)
    state = state.play(Point(2, 1))

    assert state.board[1][1] == EMPTY
    assert state.captures[Color.WHITE] == 1


def test_suicide_is_illegal() -> None:
    board = (
        (EMPTY, WHITE, EMPTY),
        (WHITE, EMPTY, WHITE),
        (EMPTY, WHITE, EMPTY),
    )
    state = GameState(size=3, board=board, to_play=Color.BLACK)

    with pytest.raises(IllegalMoveError):
        state.play(Point(1, 1))


def test_two_passes_end_game() -> None:
    state = GameState(size=9).play(None).play(None)

    assert state.is_over
    assert state.winner() is Color.WHITE
