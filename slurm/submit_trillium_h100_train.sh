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

SYGO_SLURM_LOG_DIR="${SYGO_SLURM_LOG_DIR:-${SCRATCH}/sygo-logs}"
mkdir -p "${SYGO_SLURM_LOG_DIR}"

sbatch \
  --account="${SLURM_ACCOUNT}" \
  --output="${SYGO_SLURM_LOG_DIR}/%x-%j.out" \
  --error="${SYGO_SLURM_LOG_DIR}/%x-%j.err" \
  slurm/trillium_h100_train.sbatch
