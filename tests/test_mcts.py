import random

from sygo.engine.board import GameState
from sygo.engine.mcts import (
    Evaluation,
    MCTS,
    UniformEvaluator,
    default_min_pass_moves,
    default_pass_allowed_empty_points,
    default_pass_prior_scale,
)


def test_mcts_returns_legal_policy_for_empty_9x9() -> None:
    state = GameState(size=9)
    search = MCTS(UniformEvaluator(), simulations=16, rng=random.Random(1))

    result = search.search(state)

    assert result.move_index in state.legal_move_indices()
    assert len(result.policy) == 82
    assert sum(result.policy) == 1.0
    assert sum(result.visits.values()) == 16


class PassLovingEvaluator:
    def evaluate(self, state: GameState) -> Evaluation:
        return Evaluation(priors={state.pass_index: 1.0}, value=0.0)


def test_mcts_can_suppress_opening_pass() -> None:
    state = GameState(size=9)
    search = MCTS(
        PassLovingEvaluator(),
        simulations=8,
        min_pass_moves=18,
        rng=random.Random(1),
    )

    result = search.search(state)

    assert result.move_index != state.pass_index
    assert result.policy[state.pass_index] == 0.0


def test_default_min_pass_moves_is_the_full_board_area() -> None:
    assert default_min_pass_moves(9) == 81
    assert default_min_pass_moves(13) == 169
    assert default_min_pass_moves(19) == 361


def test_default_pass_prior_scale_dampens_pass() -> None:
    assert default_pass_prior_scale() == 0.02


def test_default_pass_allowed_empty_points_scales_from_board_area() -> None:
    assert default_pass_allowed_empty_points(9) == 16
    assert default_pass_allowed_empty_points(13) == 33
    assert default_pass_allowed_empty_points(19) == 72


def test_mcts_can_dampen_pass_after_opening_gate() -> None:
    state = GameState(size=9)
    search = MCTS(
        PassLovingEvaluator(),
        simulations=8,
        min_pass_moves=0,
        pass_prior_scale=0.0,
        rng=random.Random(1),
    )

    result = search.search(state)

    assert result.move_index != state.pass_index
    assert result.policy[state.pass_index] == 0.0


def test_mcts_hides_pass_when_too_many_board_moves_remain() -> None:
    state = GameState(size=9)
    search = MCTS(
        PassLovingEvaluator(),
        simulations=8,
        min_pass_moves=999,
        pass_allowed_empty_points=80,
        rng=random.Random(1),
    )

    result = search.search(state)

    assert result.move_index != state.pass_index
    assert result.policy[state.pass_index] == 0.0


def test_mcts_allows_pass_when_few_board_moves_remain() -> None:
    state = GameState(size=9)
    search = MCTS(
        PassLovingEvaluator(),
        simulations=8,
        min_pass_moves=999,
        pass_allowed_empty_points=81,
        rng=random.Random(1),
    )

    result = search.search(state)

    assert result.policy[state.pass_index] > 0.0
