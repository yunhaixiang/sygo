#!/bin/bash
#SBATCH --job-name=sygo-9x9-h100
#SBATCH --account=YOUR_ACCOUNT_HERE
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# If your allocation requires a partition/QOS or a constraint to land on H100
# nodes, add the appropriate lines, e.g.:
#   #SBATCH --partition=...
#   #SBATCH --qos=...
#   #SBATCH --constraint=...

module purge

# Adjust these module names to match `module spider python` and `module spider cuda`
# on Trillium if needed.
module load StdEnv/2023
module load python/3.11
module load cuda/12.2

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs data checkpoints

# Recommended one-time setup on the login node before submitting:
#   python -m venv ~/venvs/sygo
#   source ~/venvs/sygo/bin/activate
#   python -m pip install --upgrade pip
#   python -m pip install -e ".[train]"
source "${SYGO_VENV:-$HOME/venvs/sygo}/bin/activate"

export PYTHONPATH=src
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "job_id=${SLURM_JOB_ID}"
echo "node_list=${SLURM_NODELIST}"
echo "submit_dir=${SLURM_SUBMIT_DIR}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
python - <<'PY'
import torch

print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_device", torch.cuda.get_device_name(0))
PY

python -m sygo.ai.training cycle \
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
