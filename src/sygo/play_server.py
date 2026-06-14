"""Local browser play server for User vs. Sygo games."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from sygo.engine.board import Color, GameState, IllegalMoveError, Point
from sygo.engine.mcts import (
    Evaluator,
    MCTS,
    UniformEvaluator,
    default_min_pass_moves,
    default_pass_allowed_empty_points,
    default_pass_prior_scale,
)


def move_label(state: GameState, move_index: int | None) -> str:
    if move_index is None or move_index == state.pass_index:
        return "pass"
    point = state.index_to_point(move_index)
    if point is None:
        return "pass"
    letters = "ABCDEFGHJKLMNOPQRSTUVWXYZ"
    return f"{letters[point.col]}{state.size - point.row}"


@dataclass
class PlaySession:
    """Mutable local game session."""

    evaluator: Evaluator
    simulations: int
    supported_sizes: list[int]
    device: str = "cpu"
    checkpoint_dir: Path = Path("checkpoints")
    current_checkpoint: str | None = None
    min_pass_moves: int | None = None
    pass_allowed_empty_points: int | None = None
    pass_prior_scale: float | None = None
    size: int = 9
    human: Color = Color.BLACK
    state: GameState = field(default_factory=lambda: GameState(size=9))
    moves: list[dict[str, Any]] = field(default_factory=list)
    model_cache: dict[str, PlayModel] = field(default_factory=dict)

    @property
    def ai(self) -> Color:
        return self.human.opponent

    @property
    def default_size(self) -> int:
        return self.supported_sizes[0]

    def config(self) -> dict[str, Any]:
        return {
            "supported_sizes": self.supported_sizes,
            "default_size": self.default_size,
            "current_checkpoint": self.current_checkpoint,
            "checkpoints": self.available_checkpoints(),
        }

    def available_checkpoints(self) -> list[dict[str, str]]:
        if not self.checkpoint_dir.exists():
            return []
        return [
            {"id": path.name, "label": path.name}
            for path in sorted(self.checkpoint_dir.glob("*.pt"))
            if path.is_file()
        ]

    def new_game(
        self,
        *,
        size: int | None = None,
        human: Color = Color.BLACK,
        checkpoint: str | None = None,
    ) -> dict[str, Any]:
        if checkpoint is not None:
            self.set_model(checkpoint)
        size = size or self.default_size
        if size not in self.supported_sizes:
            size = self.default_size
        self.size = size
        self.human = human
        self.state = GameState(size=size)
        self.moves = []
        if self.state.to_play is self.ai:
            self._play_ai_move()
        return self.payload("New game")

    def set_model(self, checkpoint_id: str | None) -> None:
        if not checkpoint_id:
            self.evaluator = UniformEvaluator()
            self.supported_sizes = [9, 13, 19]
            self.current_checkpoint = None
            return

        checkpoint_path = self._resolve_checkpoint(checkpoint_id)
        if checkpoint_path.name not in self.model_cache:
            self.model_cache[checkpoint_path.name] = build_play_model(checkpoint_path, self.device)
        play_model = self.model_cache[checkpoint_path.name]
        self.evaluator = play_model.evaluator
        self.supported_sizes = play_model.supported_sizes
        self.current_checkpoint = checkpoint_path.name

    def _resolve_checkpoint(self, checkpoint_id: str) -> Path:
        checkpoint_dir = self.checkpoint_dir.resolve()
        checkpoint_path = (checkpoint_dir / checkpoint_id).resolve()
        try:
            checkpoint_path.relative_to(checkpoint_dir)
        except ValueError as exc:
            raise ValueError("Checkpoint must be inside the configured checkpoint directory.") from exc
        if checkpoint_path.suffix != ".pt" or not checkpoint_path.is_file():
            raise ValueError(f"Unknown checkpoint: {checkpoint_id}")
        return checkpoint_path

    def play_human(self, point: Point | None) -> dict[str, Any]:
        if self.state.is_over:
            return self.payload("Game is over")
        if self.state.to_play is not self.human:
            return self.payload("Waiting for Sygo")

        played_by = self.state.to_play
        self.state = self.state.play(point)
        self._record_move(played_by, point, None)
        if not self.state.is_over:
            self._play_ai_move()
        return self.payload()

    def _play_ai_move(self) -> None:
        if self.state.is_over or self.state.to_play is not self.ai:
            return
        result = MCTS(
            evaluator=self.evaluator,
            simulations=self.simulations,
            min_pass_moves=(
                default_min_pass_moves(self.state.size)
                if self.min_pass_moves is None
                else self.min_pass_moves
            ),
            pass_allowed_empty_points=(
                default_pass_allowed_empty_points(self.state.size)
                if self.pass_allowed_empty_points is None
                else self.pass_allowed_empty_points
            ),
            pass_prior_scale=(
                default_pass_prior_scale()
                if self.pass_prior_scale is None
                else self.pass_prior_scale
            ),
            rng=random.Random(),
        ).search(self.state)
        played_by = self.state.to_play
        point = self.state.index_to_point(result.move_index)
        self.state = self.state.play(point)
        self._record_move(played_by, point, result.root_value)

    def _record_move(self, player: Color, point: Point | None, root_value: float | None) -> None:
        self.moves.append(
            {
                "number": len(self.moves) + 1,
                "player": player.value,
                "move": move_label(self.state, self.state.move_to_index(point)),
                "root_value": root_value,
            }
        )

    def payload(self, message: str | None = None) -> dict[str, Any]:
        black_player = "User" if self.human is Color.BLACK else "Sygo"
        white_player = "User" if self.human is Color.WHITE else "Sygo"
        data = {
            "mode": "play",
            "black_player": black_player,
            "white_player": white_player,
            "human": self.human.value,
            "ai": self.ai.value,
            "size": self.state.size,
            "board": [list(row) for row in self.state.board],
            "to_play": self.state.to_play.value,
            "move_count": self.state.move_count,
            "captures": {
                "black": self.state.captures[Color.BLACK],
                "white": self.state.captures[Color.WHITE],
            },
            "moves": self.moves,
            "is_over": self.state.is_over,
            "area_score": self.state.area_score() if self.state.is_over else None,
            "message": message,
        }
        if message is None:
            if self.state.is_over:
                data["message"] = f"Game over, area score B-W {self.state.area_score():.1f}"
            elif self.state.to_play is self.human:
                data["message"] = "Your move"
            else:
                data["message"] = "Sygo to play"
        return data


class PlayRequestHandler(SimpleHTTPRequestHandler):
    """HTTP handler serving both static GUI files and play API requests."""

    session: PlaySession

    def do_GET(self) -> None:
        if self.path == "/api/config":
            self._send_json(self.session.config())
            return
        if self.path == "/api/state":
            self._send_json(self.session.payload())
            return
        super().do_GET()

    def do_POST(self) -> None:
        try:
            if self.path == "/api/new":
                payload = self._read_json()
                size = int(payload.get("size", 9))
                human = Color(payload.get("human", "black"))
                checkpoint = payload.get("checkpoint")
                self._send_json(
                    self.session.new_game(size=size, human=human, checkpoint=checkpoint)
                )
            elif self.path == "/api/play":
                payload = self._read_json()
                point = None if payload.get("pass") else Point(int(payload["row"]), int(payload["col"]))
                self._send_json(self.session.play_human(point))
            else:
                self.send_error(HTTPStatus.NOT_FOUND, "Unknown API route")
        except (IllegalMoveError, KeyError, TypeError, ValueError) as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("content-length", "0"))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length))

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@dataclass(frozen=True)
class PlayModel:
    evaluator: Evaluator
    supported_sizes: list[int]


def build_play_model(checkpoint: Path | None, device: str) -> PlayModel:
    if checkpoint is None:
        return PlayModel(UniformEvaluator(), [9, 13, 19])
    from sygo.ai.policy import NeuralEvaluator
    from sygo.ai.training import load_checkpoint

    model = load_checkpoint(checkpoint, device=device)
    return PlayModel(NeuralEvaluator(model, device=device), [model.board_size])


def main() -> None:
    parser = argparse.ArgumentParser(prog="sygo-play")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument(
        "--min-pass-moves",
        type=int,
        default=None,
        help="Do not let Sygo choose pass before this move count. Defaults to board_size * board_size.",
    )
    parser.add_argument(
        "--pass-allowed-empty-points",
        type=int,
        default=None,
        help="Let Sygo pass when legal board moves are at or below this count. Defaults to 20% of board area.",
    )
    parser.add_argument(
        "--pass-prior-scale",
        type=float,
        default=None,
        help="Scale Sygo's pass prior after pass becomes legal. Defaults to 0.02.",
    )
    parser.add_argument("--directory", type=Path, default=Path.cwd())
    args = parser.parse_args()

    play_model = build_play_model(args.checkpoint, args.device)
    session = PlaySession(
        evaluator=play_model.evaluator,
        supported_sizes=play_model.supported_sizes,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        current_checkpoint=args.checkpoint.name if args.checkpoint is not None else None,
        simulations=args.simulations,
        min_pass_moves=args.min_pass_moves,
        pass_allowed_empty_points=args.pass_allowed_empty_points,
        pass_prior_scale=args.pass_prior_scale,
        size=play_model.supported_sizes[0],
        state=GameState(size=play_model.supported_sizes[0]),
    )

    class ConfiguredPlayRequestHandler(PlayRequestHandler):
        pass

    ConfiguredPlayRequestHandler.session = session

    handler = lambda *handler_args, **kwargs: ConfiguredPlayRequestHandler(  # noqa: E731
        *handler_args,
        directory=str(args.directory),
        **kwargs,
    )

    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Serving Sygo play UI at http://{args.host}:{args.port}/gui/")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
