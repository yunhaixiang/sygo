#!/bin/bash
#SBATCH --job-name=sygo-9x9-h100
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-node=h100:1

set -euo pipefail

# If your allocation requires a partition/QOS, add the appropriate lines, e.g.:
#   #SBATCH --partition=...
#   #SBATCH --qos=...

module purge

# Adjust these module names to match `module spider python` and `module spider cuda`
# on Trillium if needed.
module load StdEnv/2023
module load python/3.11
module load cuda/12.2

cd "${SLURM_SUBMIT_DIR}"

# Recommended one-time setup on the login node before submitting:
#   python -m venv ~/venvs/sygo
#   source ~/venvs/sygo/bin/activate
#   python -m pip install --upgrade pip
#   python -m pip install -e ".[train]"
SYGO_VENV="${SYGO_VENV:-$HOME/venvs/sygo}"
if [[ ! -f "${SYGO_VENV}/bin/activate" ]]; then
  echo "Sygo virtualenv not found at ${SYGO_VENV}." >&2
  echo "Create it on trig-login01 with: python -m venv ${SYGO_VENV} && source ${SYGO_VENV}/bin/activate && python -m pip install -e '.[train]'" >&2
  exit 1
fi
source "${SYGO_VENV}/bin/activate"

if [[ -z "${SYGO_RUN_ROOT:-}" ]]; then
  if [[ -n "${SCRATCH:-}" ]]; then
    SYGO_RUN_ROOT="${SCRATCH}/sygo-runs/${SLURM_JOB_ID}"
  else
    SYGO_RUN_ROOT="${SLURM_TMPDIR}/sygo-runs/${SLURM_JOB_ID}"
  fi
fi

SYGO_DATA_DIR="${SYGO_DATA_DIR:-${SYGO_RUN_ROOT}/data}"
SYGO_CHECKPOINT_DIR="${SYGO_CHECKPOINT_DIR:-${SYGO_RUN_ROOT}/checkpoints}"
SYGO_MONITOR_PATH="${SYGO_MONITOR_PATH:-${SYGO_DATA_DIR}/selfplay-monitor.json}"

mkdir -p "${SYGO_DATA_DIR}" "${SYGO_CHECKPOINT_DIR}" "$(dirname "${SYGO_MONITOR_PATH}")"

export PYTHONPATH=src
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "job_id=${SLURM_JOB_ID}"
echo "node_list=${SLURM_NODELIST}"
echo "submit_dir=${SLURM_SUBMIT_DIR}"
echo "run_root=${SYGO_RUN_ROOT}"
echo "data_dir=${SYGO_DATA_DIR}"
echo "checkpoint_dir=${SYGO_CHECKPOINT_DIR}"
echo "monitor_path=${SYGO_MONITOR_PATH}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
python - <<'PY'
import torch

print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_device", torch.cuda.get_device_name(0))
PY

python -m sygo.ai.training cycle \
  --data-dir "${SYGO_DATA_DIR}" \
  --checkpoint-dir "${SYGO_CHECKPOINT_DIR}" \
  --monitor-path "${SYGO_MONITOR_PATH}" \
  --rounds 8 \
  --games-per-round 120 \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --board-size 9 \
  --simulations 96 \
  --epochs 20 \
  --batch-size 256 \
  --channels 96 \
  --blocks 6 \
  --device cuda \
  --prefix sygo-9x9-h100-pass20 \
  --log-every 1 \
  --train-log-every 10
