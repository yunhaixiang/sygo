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

SYGO_SLURM_LOG_DIR="${SYGO_SLURM_LOG_DIR:-${SCRATCH}/sygo-logs}"
mkdir -p "${SYGO_SLURM_LOG_DIR}"

sbatch \
  --account="${SLURM_ACCOUNT}" \
  --output="${SYGO_SLURM_LOG_DIR}/%x-%j.out" \
  --error="${SYGO_SLURM_LOG_DIR}/%x-%j.err" \
  slurm/trillium_h100_train.sbatch
