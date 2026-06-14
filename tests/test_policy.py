import pytest

torch = pytest.importorskip("torch")

from sygo.ai.policy import GoResNet, NeuralEvaluator, encode_state
from sygo.engine.board import GameState


def test_go_resnet_output_shapes_for_9x9() -> None:
    model = GoResNet(board_size=9, channels=16, blocks=2)
    state = GameState(size=9)
    batch = encode_state(state).unsqueeze(0)

    outputs = model(batch)

    assert outputs["policy_logits"].shape == (1, 82)
    assert outputs["value"].shape == (1, 1)
    assert outputs["score"].shape == (1, 1)
    assert outputs["ownership"].shape == (1, 1, 9, 9)


def test_neural_evaluator_returns_legal_priors() -> None:
    model = GoResNet(board_size=9, channels=16, blocks=1)
    evaluator = NeuralEvaluator(model)
    state = GameState(size=9)

    evaluation = evaluator.evaluate(state)

    assert set(evaluation.priors) == set(state.legal_move_indices())
    assert abs(sum(evaluation.priors.values()) - 1.0) < 1e-5
    assert -1.0 <= evaluation.value <= 1.0
