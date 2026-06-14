#!/usr/bin/env python3
"""Print self-play monitor updates from Sygo JSON monitor files."""

from __future__ import annotations

import argparse
import glob
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


BLACK = 1
WHITE = 2


@dataclass(frozen=True)
class MonitorUpdate:
    path: Path
    payload: dict[str, Any]

    @property
    def key(self) -> tuple[str, float | None, int, str | None]:
        return (
            str(self.path),
            self.payload.get("updated_at"),
            int(self.payload.get("move_count") or 0),
            self.payload.get("move"),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print Sygo self-play monitor updates without the browser GUI."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help=(
            "Monitor JSON files or glob patterns. Defaults to "
            "data/selfplay-monitor*.json."
        ),
    )
    parser.add_argument("--interval", type=float, default=2.0, help="Poll interval in seconds.")
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Print one-line summaries only, without boards.",
    )
    parser.add_argument(
        "--board",
        action="store_true",
        help="Print boards after updates. This is already the default.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Print current monitor files once and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    patterns = args.paths or ["data/selfplay-monitor*.json"]
    seen: set[tuple[str, float | None, int, str | None]] = set()

    while True:
        updates = read_updates(patterns)
        if not updates:
            print(f"waiting for monitor files: {', '.join(patterns)}", flush=True)
        for update in updates:
            if update.key in seen:
                continue
            seen.add(update.key)
            print(format_update(update), flush=True)
            if not args.compact:
                print(format_board(update.payload), flush=True)
        if args.once:
            return
        time.sleep(args.interval)


def read_updates(patterns: list[str]) -> list[MonitorUpdate]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(pattern))

    updates: list[MonitorUpdate] = []
    for path in sorted(set(paths), key=sort_key):
        if not path.exists() or path.name.endswith(".tmp"):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        updates.append(MonitorUpdate(path=path, payload=payload))
    return updates


def sort_key(path: Path) -> tuple[str, int]:
    name = path.name
    marker = "-worker"
    if marker not in name:
        return name, 0
    suffix = name.split(marker, 1)[1].split(".", 1)[0]
    try:
        return name.split(marker, 1)[0], int(suffix)
    except ValueError:
        return name, 0


def format_update(update: MonitorUpdate) -> str:
    data = update.payload
    worker = format_count("worker", data.get("worker"), data.get("workers"))
    round_text = format_count("round", data.get("round"), data.get("rounds"))
    game = format_count("game", data.get("game"), data.get("games"))
    move_count = int(data.get("move_count") or 0)
    played_by = data.get("played_by") or data.get("to_play")
    move = data.get("move") or "start"
    root_value = data.get("root_value")
    value_text = f" value={root_value:.3f}" if isinstance(root_value, (int, float)) else ""
    status = "over" if data.get("is_over") else data.get("phase") or "running"
    score = data.get("area_score")
    score_text = f" score={score:.1f}" if isinstance(score, (int, float)) else ""
    source = update.path.name
    parts = [
        text
        for text in [worker, round_text, game]
        if text
    ]
    prefix = " ".join(parts)
    if prefix:
        prefix += " "
    return (
        f"{prefix}move_count={move_count} {played_by} {move} "
        f"status={status}{value_text}{score_text} file={source}"
    )


def format_count(label: str, value: Any, total: Any) -> str:
    if value is None:
        return ""
    if total is None:
        return f"{label}={value}"
    return f"{label}={value}/{total}"


def format_board(data: dict[str, Any]) -> str:
    board = data.get("board")
    if not isinstance(board, list) or not board:
        return "(no board)"
    size = int(data.get("size") or len(board))
    lines = []
    for row_index, row in enumerate(board):
        label = f"{size - row_index:2d}"
        stones = " ".join(stone_char(value) for value in row)
        lines.append(f"{label} {stones}")
    letters = "ABCDEFGHJKLMNOPQRSTUVWXYZ"[:size]
    lines.append("   " + " ".join(letters))
    return "\n".join(lines)


def stone_char(value: Any) -> str:
    if value == BLACK:
        return "X"
    if value == WHITE:
        return "O"
    return "."


if __name__ == "__main__":
    main()
