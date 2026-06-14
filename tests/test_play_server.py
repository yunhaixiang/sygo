from sygo.engine.board import Color, GameState
from sygo.engine.mcts import UniformEvaluator
from sygo.play_server import PlaySession


def make_session(**kwargs):
    return PlaySession(
        evaluator=UniformEvaluator(),
        simulations=1,
        supported_sizes=[9],
        **kwargs,
    )


def test_human_resign_sets_result_without_changing_board():
    session = make_session()

    payload = session.resign_human()

    assert payload["is_over"] is True
    assert payload["resigned_by"] == "black"
    assert payload["result"] == "W+R"
    assert payload["moves"][-1]["move"] == "resign"
    assert session.state.move_count == 0


def test_ai_resign_is_disabled_by_default():
    session = make_session(state=GameState(size=9, to_play=Color.WHITE, move_count=81))

    for _ in range(5):
        assert session._ai_should_resign(-1.0) is False


def test_ai_resign_requires_board_area_and_five_consecutive_turns():
    session = make_session(
        state=GameState(size=9, to_play=Color.WHITE, move_count=80),
        ai_resign_enabled=True,
    )

    assert session._ai_should_resign(-1.0) is False

    session.state = GameState(size=9, to_play=Color.WHITE, move_count=81)
    assert [session._ai_should_resign(-1.0) for _ in range(5)] == [
        False,
        False,
        False,
        False,
        True,
    ]


def test_ai_resign_streak_resets_when_value_recovers():
    session = make_session(
        state=GameState(size=9, to_play=Color.WHITE, move_count=81),
        ai_resign_enabled=True,
    )

    assert session._ai_should_resign(-1.0) is False
    assert session._ai_should_resign(-1.0) is False
    assert session._ai_should_resign(-0.5) is False
    assert [session._ai_should_resign(-1.0) for _ in range(5)] == [
        False,
        False,
        False,
        False,
        True,
    ]
