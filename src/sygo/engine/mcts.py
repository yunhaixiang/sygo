"""Monte Carlo Tree Search for Go positions."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Protocol

from sygo.engine.board import Color, GameState, IllegalMoveError


@dataclass(frozen=True)
class Evaluation:
    """Evaluator output from the current player's perspective."""

    priors: dict[int, float]
    value: float


class Evaluator(Protocol):
    """Callable state evaluator used by MCTS."""

    def evaluate(self, state: GameState) -> Evaluation:
        """Return move priors and value from `state.to_play` perspective."""


class UniformEvaluator:
    """A dependency-free evaluator for initial testing and random-play baselines."""

    def evaluate(self, state: GameState) -> Evaluation:
        legal = state.legal_move_indices()
        prior = 1.0 / len(legal)
        return Evaluation(priors={move: prior for move in legal}, value=0.0)


def default_min_pass_moves(board_size: int) -> int:
    """Return a conservative opening pass guard for weak early models."""

    return board_size * board_size


def default_pass_prior_scale() -> float:
    """Return the default pass-prior dampening used for weak early models."""

    return 0.02


def default_pass_allowed_empty_points(board_size: int) -> int:
    """Return how few legal board moves should remain before pass is considered."""

    return max(1, int(board_size * board_size * 0.20))


@dataclass
class Node:
    """A node in the MCTS tree."""

    state: GameState
    prior: float = 1.0
    parent: "Node | None" = None
    move_index: int | None = None
    children: dict[int, "Node"] = field(default_factory=dict)
    visits: int = 0
    value_sum: float = 0.0

    @property
    def value(self) -> float:
        return 0.0 if self.visits == 0 else self.value_sum / self.visits

    def is_expanded(self) -> bool:
        return bool(self.children)


@dataclass
class SearchResult:
    """Result of an MCTS search."""

    move_index: int
    visits: dict[int, int]
    policy: list[float]
    root_value: float


class MCTS:
    """PUCT Monte Carlo Tree Search."""

    def __init__(
        self,
        evaluator: Evaluator | None = None,
        *,
        simulations: int = 100,
        c_puct: float = 1.5,
        min_pass_moves: int = 0,
        pass_allowed_empty_points: int = 0,
        pass_prior_scale: float = 1.0,
        rng: random.Random | None = None,
    ) -> None:
        self.evaluator = evaluator or UniformEvaluator()
        self.simulations = simulations
        self.c_puct = c_puct
        self.min_pass_moves = min_pass_moves
        self.pass_allowed_empty_points = max(0, pass_allowed_empty_points)
        self.pass_prior_scale = max(0.0, pass_prior_scale)
        self.rng = rng or random.Random()

    def search(self, state: GameState, temperature: float = 0.0) -> SearchResult:
        root = Node(state=state)
        self._expand(root)

        for _ in range(self.simulations):
            node = root
            path = [node]

            while node.is_expanded() and not node.state.is_over:
                node = self._select_child(node)
                path.append(node)

            value = self._evaluate_terminal_or_expand(node)
            self._backpropagate(path, value)

        visits = {move: child.visits for move, child in root.children.items()}
        policy = self._visit_policy(state, visits, temperature)
        move_index = self._select_move_from_policy(policy)
        return SearchResult(
            move_index=move_index,
            visits=visits,
            policy=policy,
            root_value=root.value,
        )

    def _expand(self, node: Node) -> float:
        evaluation = self.evaluator.evaluate(node.state)
        legal_moves = node.state.legal_move_indices()
        non_pass_moves = [move for move in legal_moves if move != node.state.pass_index]
        if (
            node.state.pass_index in legal_moves
            and non_pass_moves
            and not self._pass_is_allowed(node.state, len(non_pass_moves))
        ):
            legal_moves = non_pass_moves
        legal = set(legal_moves)
        priors = {move: max(0.0, prior) for move, prior in evaluation.priors.items() if move in legal}
        if node.state.pass_index in priors and node.state.consecutive_passes == 0:
            priors[node.state.pass_index] *= self.pass_prior_scale

        missing = legal - set(priors)
        if missing:
            fallback = 1.0 / len(legal)
            for move in missing:
                priors[move] = fallback

        total_prior = sum(priors.values())
        if total_prior <= 0.0:
            priors = {move: 1.0 / len(legal) for move in legal}
        else:
            priors = {move: prior / total_prior for move, prior in priors.items()}

        for move, prior in priors.items():
            point = node.state.index_to_point(move)
            try:
                child_state = node.state.play(point)
            except IllegalMoveError:
                continue
            node.children[move] = Node(
                state=child_state,
                prior=prior,
                parent=node,
                move_index=move,
            )
        return evaluation.value

    def _pass_is_allowed(self, state: GameState, legal_board_moves: int) -> bool:
        if state.move_count >= self.min_pass_moves:
            return True
        return legal_board_moves <= self.pass_allowed_empty_points

    def _select_child(self, node: Node) -> Node:
        parent_visits = max(1, node.visits)

        def score(child: Node) -> float:
            q_value = -child.value
            exploration = self.c_puct * child.prior * math.sqrt(parent_visits) / (1 + child.visits)
            return q_value + exploration

        return max(node.children.values(), key=score)

    def _evaluate_terminal_or_expand(self, node: Node) -> float:
        if node.state.is_over:
            return node.state.result_value(node.state.to_play)
        return self._expand(node)

    def _backpropagate(self, path: list[Node], value: float) -> None:
        for node in reversed(path):
            node.visits += 1
            node.value_sum += value
            value = -value

    def _visit_policy(
        self, state: GameState, visits: dict[int, int], temperature: float
    ) -> list[float]:
        policy = [0.0 for _ in range(state.pass_index + 1)]
        if not visits:
            return policy

        if temperature <= 0.0:
            best_move = max(visits, key=visits.get)
            policy[best_move] = 1.0
            return policy

        scaled = {move: count ** (1.0 / temperature) for move, count in visits.items()}
        total = sum(scaled.values())
        for move, count in scaled.items():
            policy[move] = count / total
        return policy

    def _select_move_from_policy(self, policy: list[float]) -> int:
        threshold = self.rng.random()
        cumulative = 0.0
        for move, probability in enumerate(policy):
            cumulative += probability
            if threshold <= cumulative:
                return move
        return max(range(len(policy)), key=policy.__getitem__)
