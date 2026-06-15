# Sygo Weights

This directory is for selected model weights that should be tracked in git.
Generated training output under `checkpoints/` remains ignored.

## Current 9x9 Weights

- File: `sygo-9x9-h100-continue-round10.pt`
- Board size: 9x9
- Source run: Trillium Slurm job `594718`
- Training context: continued from job `594065` round 8
- Successful H100 training time recorded so far: about 15 hours

Run the GUI with this checkpoint:

```sh
PYTHONPATH=src python -u -m sygo.play_server \
  --checkpoint weights/sygo-9x9-h100-continue-round10.pt \
  --checkpoint-dir weights \
  --device cpu \
  --host 127.0.0.1 \
  --port 8000 \
  --directory .
```
