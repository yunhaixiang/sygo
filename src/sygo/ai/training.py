"""Training and self-play data utilities."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing as mp
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from sygo.ai.policy import GoResNet, NeuralEvaluator, encode_state, torch
from sygo.engine.board import BLACK, WHITE, Color, GameState
from sygo.engine.mcts import (
    MCTS,
    Evaluator,
    UniformEvaluator,
    default_min_pass_moves,
    default_pass_allowed_empty_points,
    default_pass_prior_scale,
)

SelfPlayObserver = Callable[[GameState, dict[str, Any]], None]


@dataclass(frozen=True)
class TrainingSample:
    """One neural-network training example."""

    board: list[list[int]]
    to_play: Color
    policy: list[float]
    value: float
    score: float
    ownership: list[list[float]]

    def to_json(self) -> dict[str, Any]:
        return {
            "board": self.board,
            "to_play": self.to_play.value,
            "policy": self.policy,
            "value": self.value,
            "score": self.score,
            "ownership": self.ownership,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "TrainingSample":
        return cls(
            board=data["board"],
            to_play=Color(data["to_play"]),
            policy=data["policy"],
            value=float(data["value"]),
            score=float(data["score"]),
            ownership=data["ownership"],
        )


class JsonlGoDataset(torch.utils.data.Dataset):
    """JSONL dataset for self-play or supervised samples."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.samples = [
            TrainingSample.from_json(json.loads(line))
            for line in self.path.read_text().splitlines()
            if line.strip()
        ]
        if not self.samples:
            raise ValueError(f"No training samples found in {self.path}.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        state = GameState(
            size=len(sample.board),
            board=tuple(tuple(row) for row in sample.board),
            to_play=sample.to_play,
        )
        return {
            "inputs": encode_state(state),
            "policy": torch.tensor(sample.policy, dtype=torch.float32),
            "value": torch.tensor([sample.value], dtype=torch.float32),
            "score": torch.tensor([sample.score], dtype=torch.float32),
            "ownership": torch.tensor([sample.ownership], dtype=torch.float32),
        }


@dataclass(frozen=True)
class TrainConfig:
    """Training loop configuration."""

    data: Path
    output: Path
    board_size: int = 9
    channels: int = 64
    blocks: int = 4
    batch_size: int = 64
    epochs: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cpu"
    policy_weight: float = 1.0
    value_weight: float = 1.0
    score_weight: float = 0.02
    ownership_weight: float = 0.15
    log_every: int = 10


def write_jsonl(samples: Iterable[TrainingSample], path: str | Path) -> int:
    """Write training samples to JSONL and return the number written."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample.to_json(), separators=(",", ":")))
            handle.write("\n")
            count += 1
    return count


def generate_self_play_game(
    *,
    board_size: int = 9,
    evaluator: Evaluator | None = None,
    simulations: int = 32,
    temperature: float = 1.0,
    min_pass_moves: int | None = None,
    pass_allowed_empty_points: int | None = None,
    pass_prior_scale: float | None = None,
    max_moves: int | None = None,
    rng: random.Random | None = None,
    observer: SelfPlayObserver | None = None,
    observer_context: dict[str, Any] | None = None,
) -> list[TrainingSample]:
    """Generate one self-play game as training samples."""

    rng = rng or random.Random()
    search = MCTS(
        evaluator=evaluator or UniformEvaluator(),
        simulations=simulations,
        min_pass_moves=(
            default_min_pass_moves(board_size) if min_pass_moves is None else min_pass_moves
        ),
        pass_allowed_empty_points=(
            default_pass_allowed_empty_points(board_size)
            if pass_allowed_empty_points is None
            else pass_allowed_empty_points
        ),
        pass_prior_scale=(
            default_pass_prior_scale() if pass_prior_scale is None else pass_prior_scale
        ),
        rng=rng,
    )
    state = GameState(size=board_size)
    positions: list[tuple[GameState, list[float]]] = []
    max_moves = max_moves or board_size * board_size * 2
    observer_context = observer_context or {}
    if observer is not None:
        observer(state, {**observer_context, "phase": "start", "move_index": None})

    while not state.is_over and state.move_count < max_moves:
        result = search.search(state, temperature=temperature)
        positions.append((state, result.policy))
        played_by = state.to_play
        state = state.play(state.index_to_point(result.move_index))
        if observer is not None:
            observer(
                state,
                {
                    **observer_context,
                    "phase": "move",
                    "move_index": result.move_index,
                    "played_by": played_by,
                    "root_value": result.root_value,
                },
            )

    if not state.is_over:
        played_by = state.to_play
        state = state.play(None)
        if observer is not None:
            observer(
                state,
                {
                    **observer_context,
                    "phase": "forced-pass",
                    "move_index": state.pass_index,
                    "played_by": played_by,
                },
            )
        if not state.is_over:
            played_by = state.to_play
            state = state.play(None)
            if observer is not None:
                observer(
                    state,
                    {
                        **observer_context,
                        "phase": "forced-pass",
                        "move_index": state.pass_index,
                        "played_by": played_by,
                    },
                )

    return make_samples_from_finished_game(positions, state)


def generate_self_play_games(
    *,
    games: int,
    board_size: int = 9,
    evaluator: Evaluator | None = None,
    simulations: int = 32,
    temperature: float = 1.0,
    min_pass_moves: int | None = None,
    pass_allowed_empty_points: int | None = None,
    pass_prior_scale: float | None = None,
    max_moves: int | None = None,
    rng: random.Random | None = None,
    log_every: int = 1,
    observer: SelfPlayObserver | None = None,
    observer_context: dict[str, Any] | None = None,
    game_offset: int = 0,
    total_games: int | None = None,
) -> list[TrainingSample]:
    """Generate self-play samples with progress output."""

    rng = rng or random.Random()
    samples: list[TrainingSample] = []
    started_at = time.monotonic()
    log_prefix = _self_play_log_prefix(observer_context)
    for game_number in range(1, games + 1):
        game_samples = generate_self_play_game(
            board_size=board_size,
            evaluator=evaluator,
            simulations=simulations,
            temperature=temperature,
            min_pass_moves=min_pass_moves,
            pass_allowed_empty_points=pass_allowed_empty_points,
            pass_prior_scale=pass_prior_scale,
            max_moves=max_moves,
            rng=rng,
            observer=observer,
            observer_context={
                **(observer_context or {}),
                "game": game_offset + game_number,
                "games": total_games or games,
            },
        )
        samples.extend(game_samples)
        if log_every > 0 and (game_number % log_every == 0 or game_number == games):
            elapsed = time.monotonic() - started_at
            absolute_game = game_offset + game_number
            total = total_games or games
            local_progress = []
            if game_offset != 0 or total != games:
                local_progress.append(f"local={game_number}/{games}")
            progress = " ".join(
                [
                    f"{log_prefix}self-play",
                    f"game={absolute_game}/{total}",
                    *local_progress,
                    f"samples={len(samples)}",
                    f"elapsed={elapsed:.1f}s",
                ]
            )
            print(
                progress,
                flush=True,
            )
    return samples


def _self_play_log_prefix(context: dict[str, Any] | None) -> str:
    if not context:
        return ""
    parts = []
    if context.get("round") is not None:
        rounds = f"/{context['rounds']}" if context.get("rounds") is not None else ""
        parts.append(f"round={context['round']}{rounds}")
    if context.get("worker") is not None:
        workers = f"/{context['workers']}" if context.get("workers") is not None else ""
        parts.append(f"worker={context['worker']}{workers}")
    return " ".join(parts) + (" " if parts else "")


@dataclass(frozen=True)
class SelfPlayWorkerConfig:
    """Configuration for one self-play worker process."""

    worker_id: int
    workers: int
    games: int
    game_offset: int
    total_games: int
    board_size: int
    checkpoint: Path | None
    simulations: int
    temperature: float
    min_pass_moves: int | None
    pass_allowed_empty_points: int | None
    pass_prior_scale: float | None
    max_moves: int | None
    seed: int | None
    device: str
    monitor_path: Path | None
    log_every: int
    context: dict[str, Any]


def generate_self_play_games_parallel(
    *,
    games: int,
    workers: int,
    board_size: int = 9,
    checkpoint: Path | None = None,
    simulations: int = 32,
    temperature: float = 1.0,
    min_pass_moves: int | None = None,
    pass_allowed_empty_points: int | None = None,
    pass_prior_scale: float | None = None,
    max_moves: int | None = None,
    seed: int | None = None,
    device: str = "cpu",
    monitor_path: Path | None = None,
    log_every: int = 1,
    observer_context: dict[str, Any] | None = None,
) -> list[TrainingSample]:
    """Generate self-play games across worker processes."""

    if workers <= 1 or games <= 1:
        evaluator: Evaluator | None = None
        if checkpoint is not None:
            model = load_checkpoint(checkpoint, device)
            evaluator = NeuralEvaluator(model, device=device)
        monitor = SelfPlayMonitor(monitor_path) if monitor_path is not None else None
        return generate_self_play_games(
            games=games,
            board_size=board_size,
            evaluator=evaluator,
            simulations=simulations,
            temperature=temperature,
            min_pass_moves=min_pass_moves,
            pass_allowed_empty_points=pass_allowed_empty_points,
            pass_prior_scale=pass_prior_scale,
            max_moves=max_moves,
            rng=random.Random(seed),
            log_every=log_every,
            observer=monitor,
            observer_context=observer_context,
            total_games=games,
        )

    workers = min(workers, games)
    chunks = _split_game_counts(games, workers)
    offsets: list[int] = []
    offset = 0
    for count in chunks:
        offsets.append(offset)
        offset += count

    started_at = time.monotonic()
    samples: list[TrainingSample] = []
    mp_context = mp.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp_context,
    ) as executor:
        futures = []
        for worker_id, count in enumerate(chunks, start=1):
            worker_seed = None if seed is None else seed + worker_id - 1
            futures.append(
                executor.submit(
                    _self_play_worker,
                    SelfPlayWorkerConfig(
                        worker_id=worker_id,
                        workers=workers,
                        games=count,
                        game_offset=offsets[worker_id - 1],
                        total_games=games,
                        board_size=board_size,
                        checkpoint=checkpoint,
                        simulations=simulations,
                        temperature=temperature,
                        min_pass_moves=min_pass_moves,
                        pass_allowed_empty_points=pass_allowed_empty_points,
                        pass_prior_scale=pass_prior_scale,
                        max_moves=max_moves,
                        seed=worker_seed,
                        device=device,
                        monitor_path=_worker_monitor_path(monitor_path, worker_id, workers),
                        log_every=log_every,
                        context={
                            **(observer_context or {}),
                            "worker": worker_id,
                            "workers": workers,
                        },
                    ),
                )
            )

        completed_games = 0
        for future in concurrent.futures.as_completed(futures):
            worker_id, worker_games, worker_samples = future.result()
            samples.extend(worker_samples)
            completed_games += worker_games
            elapsed = time.monotonic() - started_at
            print(
                f"self-play worker={worker_id} complete "
                f"games={completed_games}/{games} "
                f"samples={len(samples)} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    return samples


def _split_game_counts(games: int, workers: int) -> list[int]:
    base = games // workers
    remainder = games % workers
    return [base + (1 if index < remainder else 0) for index in range(workers)]


def _worker_monitor_path(path: Path | None, worker_id: int, workers: int) -> Path | None:
    if path is None:
        return None
    if workers <= 1:
        return path
    suffix = path.suffix or ".json"
    stem = path.name[: -len(path.suffix)] if path.suffix else path.name
    return path.with_name(f"{stem}-worker{worker_id}{suffix}")


def _self_play_worker(config: SelfPlayWorkerConfig) -> tuple[int, int, list[TrainingSample]]:
    try:
        torch.set_num_threads(1)
    except RuntimeError:
        pass

    evaluator: Evaluator | None = None
    if config.checkpoint is not None:
        model = load_checkpoint(config.checkpoint, config.device)
        evaluator = NeuralEvaluator(model, device=config.device)

    monitor = SelfPlayMonitor(config.monitor_path) if config.monitor_path is not None else None
    samples = generate_self_play_games(
        games=config.games,
        board_size=config.board_size,
        evaluator=evaluator,
        simulations=config.simulations,
        temperature=config.temperature,
        min_pass_moves=config.min_pass_moves,
        pass_allowed_empty_points=config.pass_allowed_empty_points,
        pass_prior_scale=config.pass_prior_scale,
        max_moves=config.max_moves,
        rng=random.Random(config.seed),
        log_every=config.log_every,
        observer=monitor,
        observer_context=config.context,
        game_offset=config.game_offset,
        total_games=config.total_games,
    )
    return config.worker_id, config.games, samples


class SelfPlayMonitor:
    """Writes the latest self-play board to a JSON file for the GUI."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.moves: list[dict[str, Any]] = []
        self.current_game_key: tuple[Any, Any] | None = None

    def __call__(self, state: GameState, context: dict[str, Any]) -> None:
        move_index = context.get("move_index")
        game_key = (context.get("round"), context.get("game"))
        if context.get("phase") == "start" or game_key != self.current_game_key:
            self.moves = []
            self.current_game_key = game_key
        move_label = _move_label(state, move_index)
        played_by = context.get("played_by")
        if move_label is not None and played_by is not None:
            self.moves.append(
                {
                    "number": len(self.moves) + 1,
                    "player": played_by.value,
                    "move": move_label,
                    "root_value": context.get("root_value"),
                    "phase": context.get("phase"),
                }
            )
        payload = {
            "updated_at": time.time(),
            "phase": context.get("phase"),
            "round": context.get("round"),
            "rounds": context.get("rounds"),
            "game": context.get("game"),
            "games": context.get("games"),
            "worker": context.get("worker"),
            "workers": context.get("workers"),
            "black_player": context.get("black_player", "Sygo"),
            "white_player": context.get("white_player", "Sygo"),
            "move_index": move_index,
            "move": move_label,
            "played_by": played_by.value if played_by is not None else None,
            "moves": self.moves,
            "root_value": context.get("root_value"),
            "size": state.size,
            "board": [list(row) for row in state.board],
            "to_play": state.to_play.value,
            "move_count": state.move_count,
            "captures": {
                "black": state.captures[Color.BLACK],
                "white": state.captures[Color.WHITE],
            },
            "is_over": state.is_over,
            "area_score": state.area_score() if state.is_over else None,
        }
        tmp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        tmp_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        tmp_path.replace(self.path)


def _move_label(state: GameState, move_index: int | None) -> str | None:
    if move_index is None:
        return None
    if move_index == state.pass_index:
        return "pass"
    point = state.index_to_point(move_index)
    if point is None:
        return "pass"
    letters = "ABCDEFGHJKLMNOPQRSTUVWXYZ"
    return f"{letters[point.col]}{state.size - point.row}"


def make_samples_from_finished_game(
    positions: list[tuple[GameState, list[float]]],
    final_state: GameState,
) -> list[TrainingSample]:
    """Turn recorded search positions into supervised network targets."""

    final_score = final_state.area_score()
    winner = final_state.winner()
    if winner is None:
        raise ValueError("Final state must be ended by two passes.")

    final_ownership = final_state.ownership_map()
    samples = []
    for state, policy in positions:
        perspective = state.to_play
        value = 1.0 if winner is perspective else -1.0
        score = final_score if perspective is Color.BLACK else -final_score
        samples.append(
            TrainingSample(
                board=[list(row) for row in state.board],
                to_play=perspective,
                policy=policy,
                value=value,
                score=score,
                ownership=_ownership_target(final_ownership, perspective),
            )
        )
    return samples


def _ownership_target(
    ownership: tuple[tuple[int, ...], ...],
    perspective: Color,
) -> list[list[float]]:
    target = []
    current = perspective.stone
    opponent = perspective.opponent.stone
    for row in ownership:
        target_row = []
        for owner in row:
            if owner == current:
                target_row.append(1.0)
            elif owner == opponent:
                target_row.append(-1.0)
            else:
                target_row.append(0.0)
        target.append(target_row)
    return target


def train_model(config: TrainConfig) -> dict[str, float]:
    """Train a `GoResNet` from JSONL samples and save a checkpoint."""

    device = torch.device(config.device)
    dataset = JsonlGoDataset(config.data)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )
    model = GoResNet(
        board_size=config.board_size,
        channels=config.channels,
        blocks=config.blocks,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    metrics: dict[str, float] = {}
    for epoch in range(config.epochs):
        metrics = _train_epoch(model, loader, optimizer, config, device, epoch + 1)
        print(
            f"epoch={epoch + 1} "
            f"loss={metrics['loss']:.4f} "
            f"policy={metrics['policy_loss']:.4f} "
            f"value={metrics['value_loss']:.4f} "
            f"score={metrics['score_loss']:.4f} "
            f"ownership={metrics['ownership_loss']:.4f}"
        )

    config.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": {
                "board_size": config.board_size,
                "channels": config.channels,
                "blocks": config.blocks,
            },
            "metrics": metrics,
        },
        config.output,
    )
    return metrics


def _train_epoch(
    model: GoResNet,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    model.train()
    totals = {
        "loss": 0.0,
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "score_loss": 0.0,
        "ownership_loss": 0.0,
    }
    batches = 0

    total_batches = len(loader)
    started_at = time.monotonic()

    for batch_index, batch in enumerate(loader, start=1):
        inputs = batch["inputs"].to(device)
        policy_target = batch["policy"].to(device)
        value_target = batch["value"].to(device)
        score_target = batch["score"].to(device)
        ownership_target = batch["ownership"].to(device)

        outputs = model(inputs)
        policy_loss = _soft_policy_cross_entropy(outputs["policy_logits"], policy_target)
        value_loss = torch.nn.functional.mse_loss(outputs["value"], value_target)
        score_loss = torch.nn.functional.mse_loss(outputs["score"], score_target)
        ownership_loss = torch.nn.functional.mse_loss(outputs["ownership"], ownership_target)
        loss = (
            config.policy_weight * policy_loss
            + config.value_weight * value_loss
            + config.score_weight * score_loss
            + config.ownership_weight * ownership_loss
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_metrics = {
            "loss": loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "score_loss": score_loss,
            "ownership_loss": ownership_loss,
        }
        for key, value in batch_metrics.items():
            totals[key] += float(value.detach().cpu())
        batches += 1

        if config.log_every > 0 and (
            batch_index % config.log_every == 0 or batch_index == total_batches
        ):
            elapsed = time.monotonic() - started_at
            print(
                f"epoch={epoch} "
                f"batch={batch_index}/{total_batches} "
                f"loss={float(loss.detach().cpu()):.4f} "
                f"policy={float(policy_loss.detach().cpu()):.4f} "
                f"value={float(value_loss.detach().cpu()):.4f} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    return {key: value / max(1, batches) for key, value in totals.items()}


def _soft_policy_cross_entropy(
    logits: torch.Tensor,
    target_distribution: torch.Tensor,
) -> torch.Tensor:
    log_probabilities = torch.nn.functional.log_softmax(logits, dim=1)
    return -(target_distribution * log_probabilities).sum(dim=1).mean()


def load_checkpoint(path: str | Path, device: str = "cpu") -> GoResNet:
    """Load a saved `GoResNet` checkpoint."""

    checkpoint = torch.load(path, map_location=device)
    config = checkpoint["config"]
    model = GoResNet(**config)
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device)


@dataclass(frozen=True)
class CycleConfig:
    """Automated self-play/training loop configuration."""

    data_dir: Path = Path("data")
    checkpoint_dir: Path = Path("checkpoints")
    initial_checkpoint: Path | None = None
    rounds: int = 3
    games_per_round: int = 10
    board_size: int = 9
    simulations: int = 32
    temperature: float = 1.0
    min_pass_moves: int | None = None
    pass_allowed_empty_points: int | None = None
    pass_prior_scale: float | None = None
    max_moves: int | None = None
    channels: int = 64
    blocks: int = 4
    batch_size: int = 64
    epochs: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cpu"
    seed: int | None = None
    log_every: int = 1
    train_log_every: int = 10
    prefix: str = "sygo-9x9"
    monitor_path: Path | None = Path("data/selfplay-monitor.json")
    workers: int = 1


def run_training_cycle(config: CycleConfig) -> Path:
    """Run repeated self-play then training rounds.

    Returns the final checkpoint path.
    """

    config.data_dir.mkdir(parents=True, exist_ok=True)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(config.seed)
    current_checkpoint = config.initial_checkpoint
    monitor = SelfPlayMonitor(config.monitor_path) if config.monitor_path is not None else None

    for round_number in range(1, config.rounds + 1):
        round_started = time.monotonic()
        data_path = config.data_dir / f"{config.prefix}-round{round_number}.jsonl"
        checkpoint_path = config.checkpoint_dir / f"{config.prefix}-round{round_number}.pt"

        evaluator: Evaluator | None = None
        if current_checkpoint is not None:
            print(f"round={round_number} loading checkpoint={current_checkpoint}", flush=True)
            if config.workers <= 1:
                model = load_checkpoint(current_checkpoint, config.device)
                evaluator = NeuralEvaluator(model, device=config.device)
        else:
            print(f"round={round_number} using uniform evaluator", flush=True)

        if config.workers > 1:
            samples = generate_self_play_games_parallel(
                games=config.games_per_round,
                workers=config.workers,
                board_size=config.board_size,
                checkpoint=current_checkpoint,
                simulations=config.simulations,
                temperature=config.temperature,
                min_pass_moves=config.min_pass_moves,
                pass_allowed_empty_points=config.pass_allowed_empty_points,
                pass_prior_scale=config.pass_prior_scale,
                max_moves=config.max_moves,
                seed=None if config.seed is None else config.seed + round_number * 100_000,
                device=config.device,
                monitor_path=config.monitor_path,
                log_every=config.log_every,
                observer_context={
                    "round": round_number,
                    "rounds": config.rounds,
                },
            )
        else:
            samples = generate_self_play_games(
                games=config.games_per_round,
                board_size=config.board_size,
                evaluator=evaluator,
                simulations=config.simulations,
                temperature=config.temperature,
                min_pass_moves=config.min_pass_moves,
                pass_allowed_empty_points=config.pass_allowed_empty_points,
                pass_prior_scale=config.pass_prior_scale,
                max_moves=config.max_moves,
                rng=rng,
                log_every=config.log_every,
                observer=monitor,
                observer_context={
                    "round": round_number,
                    "rounds": config.rounds,
                },
            )
        sample_count = write_jsonl(samples, data_path)
        print(
            f"round={round_number} wrote samples={sample_count} data={data_path}",
            flush=True,
        )

        train_model(
            TrainConfig(
                data=data_path,
                output=checkpoint_path,
                board_size=config.board_size,
                channels=config.channels,
                blocks=config.blocks,
                batch_size=config.batch_size,
                epochs=config.epochs,
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
                device=config.device,
                log_every=config.train_log_every,
            )
        )
        current_checkpoint = checkpoint_path
        elapsed = time.monotonic() - round_started
        print(
            f"round={round_number} complete checkpoint={checkpoint_path} elapsed={elapsed:.1f}s",
            flush=True,
        )

    if current_checkpoint is None:
        raise RuntimeError("Training cycle did not produce a checkpoint.")
    return current_checkpoint


def train() -> None:
    """CLI entry point for training and self-play sample generation."""

    parser = argparse.ArgumentParser(prog="sygo-train")
    subcommands = parser.add_subparsers(dest="command", required=True)

    generate_parser = subcommands.add_parser("self-play")
    generate_parser.add_argument("--output", type=Path, required=True)
    generate_parser.add_argument("--games", type=int, default=1)
    generate_parser.add_argument("--board-size", type=int, default=9)
    generate_parser.add_argument("--simulations", type=int, default=32)
    generate_parser.add_argument("--temperature", type=float, default=1.0)
    generate_parser.add_argument(
        "--min-pass-moves",
        type=int,
        default=None,
        help="Do not allow pass before this move count during self-play. Defaults to board_size * board_size.",
    )
    generate_parser.add_argument(
        "--pass-allowed-empty-points",
        type=int,
        default=None,
        help="Allow pass when legal board moves are at or below this count. Defaults to 20% of board area.",
    )
    generate_parser.add_argument(
        "--pass-prior-scale",
        type=float,
        default=None,
        help="Scale the pass prior after pass becomes legal. Defaults to 0.02.",
    )
    generate_parser.add_argument("--max-moves", type=int, default=None)
    generate_parser.add_argument("--seed", type=int, default=None)
    generate_parser.add_argument("--checkpoint", type=Path, default=None)
    generate_parser.add_argument("--device", default="cpu")
    generate_parser.add_argument("--log-every", type=int, default=1)
    generate_parser.add_argument("--monitor-path", type=Path, default=Path("data/selfplay-monitor.json"))
    generate_parser.add_argument("--workers", type=int, default=1)

    train_parser = subcommands.add_parser("fit")
    train_parser.add_argument("--data", type=Path, required=True)
    train_parser.add_argument("--output", type=Path, required=True)
    train_parser.add_argument("--board-size", type=int, default=9)
    train_parser.add_argument("--channels", type=int, default=64)
    train_parser.add_argument("--blocks", type=int, default=4)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--epochs", type=int, default=1)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--device", default="cpu")
    train_parser.add_argument("--log-every", type=int, default=10)

    cycle_parser = subcommands.add_parser("cycle")
    cycle_parser.add_argument("--data-dir", type=Path, default=Path("data"))
    cycle_parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    cycle_parser.add_argument("--initial-checkpoint", type=Path, default=None)
    cycle_parser.add_argument("--rounds", type=int, default=3)
    cycle_parser.add_argument("--games-per-round", type=int, default=10)
    cycle_parser.add_argument("--board-size", type=int, default=9)
    cycle_parser.add_argument("--simulations", type=int, default=32)
    cycle_parser.add_argument("--temperature", type=float, default=1.0)
    cycle_parser.add_argument(
        "--min-pass-moves",
        type=int,
        default=None,
        help="Do not allow pass before this move count during self-play. Defaults to board_size * board_size.",
    )
    cycle_parser.add_argument(
        "--pass-allowed-empty-points",
        type=int,
        default=None,
        help="Allow pass when legal board moves are at or below this count. Defaults to 20% of board area.",
    )
    cycle_parser.add_argument(
        "--pass-prior-scale",
        type=float,
        default=None,
        help="Scale the pass prior after pass becomes legal. Defaults to 0.02.",
    )
    cycle_parser.add_argument("--max-moves", type=int, default=None)
    cycle_parser.add_argument("--channels", type=int, default=64)
    cycle_parser.add_argument("--blocks", type=int, default=4)
    cycle_parser.add_argument("--batch-size", type=int, default=64)
    cycle_parser.add_argument("--epochs", type=int, default=5)
    cycle_parser.add_argument("--learning-rate", type=float, default=1e-3)
    cycle_parser.add_argument("--weight-decay", type=float, default=1e-4)
    cycle_parser.add_argument("--device", default="cpu")
    cycle_parser.add_argument("--seed", type=int, default=None)
    cycle_parser.add_argument("--log-every", type=int, default=1)
    cycle_parser.add_argument("--train-log-every", type=int, default=10)
    cycle_parser.add_argument("--prefix", default="sygo-9x9")
    cycle_parser.add_argument("--monitor-path", type=Path, default=Path("data/selfplay-monitor.json"))
    cycle_parser.add_argument("--workers", type=int, default=1)

    args = parser.parse_args()
    if args.command == "self-play":
        if args.workers > 1:
            samples = generate_self_play_games_parallel(
                games=args.games,
                workers=args.workers,
                board_size=args.board_size,
                checkpoint=args.checkpoint,
                simulations=args.simulations,
                temperature=args.temperature,
                min_pass_moves=args.min_pass_moves,
                pass_allowed_empty_points=args.pass_allowed_empty_points,
                pass_prior_scale=args.pass_prior_scale,
                max_moves=args.max_moves,
                seed=args.seed,
                device=args.device,
                monitor_path=args.monitor_path,
                log_every=args.log_every,
            )
        else:
            rng = random.Random(args.seed)
            evaluator: Evaluator | None = None
            if args.checkpoint is not None:
                model = load_checkpoint(args.checkpoint, args.device)
                evaluator = NeuralEvaluator(model, device=args.device)
            monitor = SelfPlayMonitor(args.monitor_path) if args.monitor_path is not None else None
            samples = generate_self_play_games(
                games=args.games,
                board_size=args.board_size,
                evaluator=evaluator,
                simulations=args.simulations,
                temperature=args.temperature,
                min_pass_moves=args.min_pass_moves,
                pass_allowed_empty_points=args.pass_allowed_empty_points,
                pass_prior_scale=args.pass_prior_scale,
                max_moves=args.max_moves,
                rng=rng,
                log_every=args.log_every,
                observer=monitor,
            )
        count = write_jsonl(samples, args.output)
        print(f"wrote {count} samples to {args.output}")
    elif args.command == "fit":
        train_model(
            TrainConfig(
                data=args.data,
                output=args.output,
                board_size=args.board_size,
                channels=args.channels,
                blocks=args.blocks,
                batch_size=args.batch_size,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                device=args.device,
                log_every=args.log_every,
            )
        )
    elif args.command == "cycle":
        final_checkpoint = run_training_cycle(
            CycleConfig(
                data_dir=args.data_dir,
                checkpoint_dir=args.checkpoint_dir,
                initial_checkpoint=args.initial_checkpoint,
                rounds=args.rounds,
                games_per_round=args.games_per_round,
                board_size=args.board_size,
                simulations=args.simulations,
                temperature=args.temperature,
                min_pass_moves=args.min_pass_moves,
                pass_allowed_empty_points=args.pass_allowed_empty_points,
                pass_prior_scale=args.pass_prior_scale,
                max_moves=args.max_moves,
                channels=args.channels,
                blocks=args.blocks,
                batch_size=args.batch_size,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                device=args.device,
                seed=args.seed,
                log_every=args.log_every,
                train_log_every=args.train_log_every,
                prefix=args.prefix,
                monitor_path=args.monitor_path,
                workers=args.workers,
            )
        )
        print(f"final checkpoint={final_checkpoint}")


if __name__ == "__main__":
    train()
