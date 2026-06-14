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

sbatch --account="${SLURM_ACCOUNT}" slurm/trillium_h100_train.sbatch
