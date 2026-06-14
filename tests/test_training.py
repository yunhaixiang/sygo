import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from sygo.ai.training import (
    CycleConfig,
    JsonlGoDataset,
    SelfPlayMonitor,
    TrainConfig,
    _worker_monitor_path,
    generate_self_play_games_parallel,
    generate_self_play_games,
    generate_self_play_game,
    run_training_cycle,
    train_model,
    write_jsonl,
)


def test_self_play_samples_can_train_tiny_model(tmp_path: Path) -> None:
    samples = generate_self_play_game(simulations=1, max_moves=2)
    data_path = tmp_path / "samples.jsonl"
    checkpoint_path = tmp_path / "checkpoint.pt"

    assert write_jsonl(samples, data_path) == len(samples)
    assert len(JsonlGoDataset(data_path)) == len(samples)

    metrics = train_model(
        TrainConfig(
            data=data_path,
            output=checkpoint_path,
            channels=8,
            blocks=1,
            batch_size=2,
            epochs=1,
        )
    )

    assert checkpoint_path.exists()
    assert metrics["loss"] > 0


def test_self_play_games_progress_helper_returns_samples() -> None:
    samples = generate_self_play_games(games=1, simulations=1, max_moves=1, log_every=0)

    assert len(samples) == 1


def test_parallel_self_play_returns_all_samples() -> None:
    samples = generate_self_play_games_parallel(
        games=2,
        workers=2,
        simulations=1,
        max_moves=1,
        seed=1,
        monitor_path=None,
    )

    assert len(samples) == 2


def test_worker_monitor_path_uses_worker_suffix() -> None:
    assert _worker_monitor_path(Path("data/selfplay-monitor.json"), 2, 4) == Path(
        "data/selfplay-monitor-worker2.json"
    )
    assert _worker_monitor_path(Path("data/selfplay-monitor.json"), 1, 1) == Path(
        "data/selfplay-monitor.json"
    )


def test_self_play_monitor_writes_latest_board(tmp_path: Path) -> None:
    monitor_path = tmp_path / "monitor.json"
    samples = generate_self_play_games(
        games=1,
        simulations=1,
        max_moves=1,
        log_every=0,
        observer=SelfPlayMonitor(monitor_path),
    )
    data = json.loads(monitor_path.read_text())

    assert len(samples) == 1
    assert data["size"] == 9
    assert len(data["board"]) == 9
    assert data["move_count"] >= 1
    assert data["moves"][0]["player"] == "black"


def test_training_cycle_writes_round_outputs(tmp_path: Path) -> None:
    checkpoint = run_training_cycle(
        CycleConfig(
            data_dir=tmp_path / "data",
            checkpoint_dir=tmp_path / "checkpoints",
            rounds=1,
            games_per_round=1,
            simulations=1,
            max_moves=1,
            channels=8,
            blocks=1,
            batch_size=1,
            epochs=1,
            log_every=0,
            train_log_every=0,
        )
    )

    assert checkpoint.exists()
    assert (tmp_path / "data" / "sygo-9x9-round1.jsonl").exists()
