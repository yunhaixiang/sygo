# Sygo

Sygo is a from-scratch Go AI project scaffold. The intended direction is:

- a rules engine that can support serious game analysis and self-play
- GTP support for engine integration and online computer Go servers
- SGF parsing/writing for datasets, evaluation, and review
- PyTorch training pipelines
- optional neurosymbolic experiments layered around the core engine
- a local browser GUI for play, inspection, and future AI interaction

## Rules Target

Sygo targets online computer Go compatibility first. The default ruleset is
CGOS-style modified Tromp-Taylor:

- area scoring
- suicide forbidden
- full play-out for unresolved dead stones or scoring disputes
- komi: 7.5 on 19x19 and 13x13, 7.0 on 9x9
- simple ko initially, with superko support planned as an engine option

The GUI currently enforces stone placement, capture, pass, and no-suicide. Ko,
final area scoring, and full GTP/SGF rule metadata are still scaffold items.

## Layout

```text
src/sygo/
  engine/         Go rules, GTP, and SGF integration points
  ai/             PyTorch models, training, and experiments
gui/              Static browser GUI
tests/            Unit tests
```

## GUI

Open `gui/index.html` in a browser, or serve the repository root:

```bash
python -m http.server 8000
```

Then visit `http://localhost:8000/gui/`.

The initial GUI supports 9x9, 13x13, and 19x19 board sizes, stone placement,
captures, pass, undo, reset, and a move log.

Play against a trained checkpoint:

```bash
PYTHONPATH=src python -m sygo.play_server --checkpoint checkpoints/sygo-9x9-round1.pt --port 8000 --simulations 64
```

Then open `http://127.0.0.1:8000/gui/` and click `Play Sygo`.
The play server enables the board size supported by the loaded checkpoint. The
example checkpoint above is 9x9; future 13x13 or 19x19 checkpoints will expose
their own board size to the GUI.

The play server suppresses early pass choices because tiny early models can
overrate pass. Override the hard opening gate with `--min-pass-moves`; the
default is the full board area, `board_size * board_size`. Sygo can also pass
when the number of legal board moves is small enough; override that with
`--pass-allowed-empty-points`, defaulting to 20% of board area. After pass is
allowed, Sygo still dampens pass with `--pass-prior-scale`, default `0.02`.

## Python Package

Install in editable mode once dependencies are ready:

```bash
python -m pip install -e ".[dev]"
```

PyTorch is listed as an optional `train` dependency so CPU/GPU builds can be selected deliberately.

## First 9x9 AI Components

The initial AI implementation includes:

- `GameState`: legal play, capture, no-suicide, pass, simple ko, and area scoring
- `MCTS`: PUCT search with a uniform baseline evaluator
- `GoResNet`: PyTorch policy/value/score/ownership network for fixed-size boards
- `NeuralEvaluator`: adapter that lets MCTS use `GoResNet`

Run a small dependency-free MCTS search:

```bash
PYTHONPATH=src python -c "from sygo.engine.board import GameState; from sygo.engine.mcts import MCTS; s=GameState(size=9); r=MCTS(simulations=32).search(s); print(r.move_index, r.root_value)"
```

Run a neural model smoke test after installing PyTorch:

```bash
PYTHONPATH=src python -c "from sygo.ai.policy import GoResNet, encode_state; from sygo.engine.board import GameState; m=GoResNet(board_size=9); print(m(encode_state(GameState(size=9)).unsqueeze(0))['policy_logits'].shape)"
```

Generate a tiny self-play dataset:

```bash
PYTHONPATH=src python -m sygo.ai.training self-play --output data/selfplay-9x9.jsonl --games 10 --workers 4 --board-size 9 --simulations 32 --log-every 1
```

Self-play also uses the pass controls by default. Override the hard gate with
`--min-pass-moves`, override the legal-board-move threshold with
`--pass-allowed-empty-points`, and override the soft pass dampening with
`--pass-prior-scale`. Use
`--min-pass-moves 0 --pass-allowed-empty-points 100000 --pass-prior-scale 1.0`
to allow pass immediately without dampening.

Train a first 9x9 checkpoint:

```bash
PYTHONPATH=src python -m sygo.ai.training fit --data data/selfplay-9x9.jsonl --output checkpoints/sygo-9x9.pt --board-size 9 --epochs 5 --log-every 10
```

Run automated self-play/training rounds:

```bash
PYTHONPATH=src python -m sygo.ai.training cycle --rounds 3 --games-per-round 20 --workers 4 --board-size 9 --simulations 64 --epochs 5 --log-every 1 --train-log-every 10
```

The cycle command writes round data and checkpoints like:

```text
data/sygo-9x9-round1.jsonl
checkpoints/sygo-9x9-round1.pt
data/sygo-9x9-round2.jsonl
checkpoints/sygo-9x9-round2.pt
```

To watch self-play on the GUI board while it runs, serve the repo root:

```bash
python -m http.server 8000
```

Open `http://localhost:8000/gui/`, click `Watch self-play`, then run self-play or a cycle. The training code writes the latest board to:

```text
data/selfplay-monitor.json
```

With `--workers > 1`, each worker writes a separate monitor file:

```text
data/selfplay-monitor-worker1.json
data/selfplay-monitor-worker2.json
...
```

Use the GUI `Worker` selector to choose which self-play worker to watch. You can choose a different monitor base path, but the GUI currently watches the default `data/selfplay-monitor*.json` paths:

```bash
PYTHONPATH=src python -m sygo.ai.training cycle --rounds 3 --games-per-round 20 --monitor-path data/selfplay-monitor.json
```

On a cluster, you can print monitor updates and ASCII boards directly in the terminal without running the browser GUI:

```bash
python scripts/monitor_selfplay.py "$SCRATCH/sygo-runs/594065/data/selfplay-monitor*.json"
```

Use `--compact` for one-line summaries without boards:

```bash
python scripts/monitor_selfplay.py --compact "$SCRATCH/sygo-runs/594065/data/selfplay-monitor*.json"
```
