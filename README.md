# Sygo

Sygo is a from-scratch Go AI project scaffold. The intended direction is:

- a rules engine that can support serious game analysis and self-play
- GTP support for engine integration and online computer Go servers
- SGF parsing/writing for datasets, evaluation, and review
- PyTorch training pipelines
- optional neurosymbolic experiments layered around the core engine
- a local browser GUI for play, inspection, and future AI interaction

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

The initial GUI supports 9x9, 13x13, and 19x19 board sizes, stone placement, captures, pass, undo, reset, and a move log.

## Python Package

Install in editable mode once dependencies are ready:

```bash
python -m pip install -e ".[dev]"
```

PyTorch is listed as an optional `train` dependency so CPU/GPU builds can be selected deliberately.
