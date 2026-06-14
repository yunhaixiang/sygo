#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

: "${SLURM_ACCOUNT:?Set SLURM_ACCOUNT before submitting, e.g. export SLURM_ACCOUNT=def-yourpi or put SLURM_ACCOUNT=def-yourpi in .env}"

if [[ "$(hostname -s)" != "trig-login01" ]]; then
  echo "Submit H100 jobs from trig-login01, e.g. ssh trig-login01, then rerun this script." >&2
  exit 1
fi

: "${SCRATCH:?SCRATCH is not set. Trillium expects job logs and generated data under scratch.}"

SYGO_VENV="${SYGO_VENV:-$HOME/venvs/sygo}"
if [[ ! -f "${SYGO_VENV}/bin/activate" ]]; then
  cat >&2 <<EOF
Sygo virtualenv not found at:
  ${SYGO_VENV}

Run this one-time setup on trig-login01 from the sygo repo:

  module purge
  module load StdEnv/2023
  module load python/3.11
  python -m venv "${SYGO_VENV}"
  source "${SYGO_VENV}/bin/activate"
  python -m pip install --upgrade pip
  python -m pip install -e ".[train]"

Then rerun:
  bash slurm/submit_trillium_h100_train.sh

To use a different venv path, set SYGO_VENV in .env.
EOF
  exit 1
fi
export SYGO_VENV

SYGO_CPUS_PER_TASK="${SYGO_CPUS_PER_TASK:-24}"
SYGO_TIME="${SYGO_TIME:-12:00:00}"
SYGO_ROUNDS="${SYGO_ROUNDS:-10}"
SYGO_GAMES_PER_ROUND="${SYGO_GAMES_PER_ROUND:-240}"
SYGO_WORKERS="${SYGO_WORKERS:-${SYGO_CPUS_PER_TASK}}"
SYGO_BOARD_SIZE="${SYGO_BOARD_SIZE:-9}"
SYGO_SIMULATIONS="${SYGO_SIMULATIONS:-128}"
SYGO_EPOCHS="${SYGO_EPOCHS:-24}"
SYGO_BATCH_SIZE="${SYGO_BATCH_SIZE:-256}"
SYGO_CHANNELS="${SYGO_CHANNELS:-96}"
SYGO_BLOCKS="${SYGO_BLOCKS:-6}"
SYGO_DEVICE="${SYGO_DEVICE:-cuda}"
SYGO_PREFIX="${SYGO_PREFIX:-sygo-9x9-h100-continue}"
SYGO_LOG_EVERY="${SYGO_LOG_EVERY:-1}"
SYGO_TRAIN_LOG_EVERY="${SYGO_TRAIN_LOG_EVERY:-10}"

export \
  SYGO_CPUS_PER_TASK \
  SYGO_TIME \
  SYGO_ROUNDS \
  SYGO_GAMES_PER_ROUND \
  SYGO_WORKERS \
  SYGO_BOARD_SIZE \
  SYGO_SIMULATIONS \
  SYGO_EPOCHS \
  SYGO_BATCH_SIZE \
  SYGO_CHANNELS \
  SYGO_BLOCKS \
  SYGO_DEVICE \
  SYGO_PREFIX \
  SYGO_LOG_EVERY \
  SYGO_TRAIN_LOG_EVERY \
  SYGO_INITIAL_CHECKPOINT

SYGO_SLURM_LOG_DIR="${SYGO_SLURM_LOG_DIR:-${SCRATCH}/sygo-logs}"
mkdir -p "${SYGO_SLURM_LOG_DIR}"

sbatch \
  --account="${SLURM_ACCOUNT}" \
  --time="${SYGO_TIME}" \
  --cpus-per-task="${SYGO_CPUS_PER_TASK}" \
  --output="${SYGO_SLURM_LOG_DIR}/%x-%j.out" \
  --error="${SYGO_SLURM_LOG_DIR}/%x-%j.err" \
  --export=ALL,SYGO_VENV,SYGO_INITIAL_CHECKPOINT,SYGO_ROUNDS,SYGO_GAMES_PER_ROUND,SYGO_WORKERS,SYGO_BOARD_SIZE,SYGO_SIMULATIONS,SYGO_EPOCHS,SYGO_BATCH_SIZE,SYGO_CHANNELS,SYGO_BLOCKS,SYGO_DEVICE,SYGO_PREFIX,SYGO_LOG_EVERY,SYGO_TRAIN_LOG_EVERY \
  slurm/trillium_h100_train.sbatch
