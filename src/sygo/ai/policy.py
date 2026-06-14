"""PyTorch ResNet policy/value model for 9x9 Go."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sygo.engine.board import EMPTY, Color, GameState
from sygo.engine.mcts import Evaluation

if TYPE_CHECKING:
    import torch


def _load_torch() -> tuple[object, object, object]:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for sygo.ai.policy. Install with `python -m pip install -e '.[train]'`."
        ) from exc
    return torch, nn, functional


torch, nn, F = _load_torch()


class ResidualBlock(nn.Module):
    """Two-convolution residual block."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(channels)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        residual = x
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return F.relu(x + residual)


class GoResNet(nn.Module):
    """Small AlphaZero/KataGo-style network for fixed-size Go boards."""

    def __init__(
        self,
        *,
        board_size: int = 9,
        input_channels: int = 5,
        channels: int = 64,
        blocks: int = 4,
    ) -> None:
        super().__init__()
        self.board_size = board_size
        self.input_channels = input_channels
        self.move_count = board_size * board_size + 1

        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.trunk = nn.Sequential(*(ResidualBlock(channels) for _ in range(blocks)))

        points = board_size * board_size
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * points, self.move_count),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(points, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )
        self.score_head = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(points, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.ownership_head = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, x: "torch.Tensor") -> dict[str, "torch.Tensor"]:
        features = self.trunk(self.stem(x))
        return {
            "policy_logits": self.policy_head(features),
            "value": self.value_head(features),
            "score": self.score_head(features),
            "ownership": torch.tanh(self.ownership_head(features)),
        }


PolicyValueModel = GoResNet


def encode_state(state: GameState) -> "torch.Tensor":
    """Encode a state as `[5, size, size]` from the current player's perspective."""

    board = torch.zeros((5, state.size, state.size), dtype=torch.float32)
    current_stone = state.to_play.stone
    opponent_stone = state.to_play.opponent.stone

    for row in range(state.size):
        for col in range(state.size):
            stone = state.board[row][col]
            if stone == current_stone:
                board[0, row, col] = 1.0
            elif stone == opponent_stone:
                board[1, row, col] = 1.0
            if stone == EMPTY:
                board[2, row, col] = 1.0

    if state.to_play is Color.BLACK:
        board[3].fill_(1.0)
    else:
        board[4].fill_(1.0)
    return board


class NeuralEvaluator:
    """MCTS evaluator backed by a `GoResNet` model."""

    def __init__(self, model: GoResNet, device: str | None = None) -> None:
        self.model = model
        self.device = torch.device(device) if device else next(model.parameters()).device

    def evaluate(self, state: GameState) -> Evaluation:
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            encoded = encode_state(state).unsqueeze(0).to(self.device)
            outputs = self.model(encoded)
            logits = outputs["policy_logits"][0]
            legal = state.legal_move_indices()
            mask = torch.full_like(logits, float("-inf"))
            mask[legal] = logits[legal]
            probabilities = torch.softmax(mask, dim=0).detach().cpu()
            value = float(outputs["value"][0, 0].detach().cpu())

        if was_training:
            self.model.train()

        priors = {move: float(probabilities[move]) for move in legal}
        return Evaluation(priors=priors, value=value)
