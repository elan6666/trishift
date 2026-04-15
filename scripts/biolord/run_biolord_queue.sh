#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONDA_BASE="${CONDA_BASE:-/root/miniconda3}"
CONDA_SH="${CONDA_BASE}/etc/profile.d/conda.sh"
ENV_NAME="${BIOLORD_ENV_NAME:-biolord}"
PYTHON_BIN="${PYTHON_BIN:-}"

if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ ! -f "${CONDA_SH}" ]]; then
    echo "[biolord-queue] missing conda activation script: ${CONDA_SH}" >&2
    exit 1
  fi
  # shellcheck source=/dev/null
  source "${CONDA_SH}"
  conda activate "${ENV_NAME}"
  PYTHON_BIN="$(command -v python)"
fi

cd "${REPO_ROOT}"
mkdir -p artifacts/results/biolord

declare -a TASKS=(
  "adamson|scripts/biolord/adamson/run_biolord_adamson.py"
  "norman|scripts/biolord/norman/run_biolord_norman.py"
  "dixit|scripts/biolord/dixit/run_biolord_dixit.py"
)

echo "[biolord-queue] repo_root=${REPO_ROOT}"
echo "[biolord-queue] python=${PYTHON_BIN}"
echo "[biolord-queue] started_at=$(date '+%Y-%m-%d %H:%M:%S')"

for item in "${TASKS[@]}"; do
  IFS="|" read -r dataset script_path <<< "${item}"
  if [[ ! -f "${script_path}" ]]; then
    echo "[biolord-queue] missing script for ${dataset}: ${script_path}" >&2
    exit 1
  fi
  config_path="$(dirname "${script_path}")/config.yaml"
  if [[ ! -f "${config_path}" ]]; then
    echo "[biolord-queue] missing config for ${dataset}: ${config_path}" >&2
    exit 1
  fi

  echo "[biolord-queue] dataset=${dataset} status=starting at $(date '+%Y-%m-%d %H:%M:%S')"
  "${PYTHON_BIN}" "${script_path}"
  echo "[biolord-queue] dataset=${dataset} status=finished at $(date '+%Y-%m-%d %H:%M:%S')"
done

echo "[biolord-queue] all_done at $(date '+%Y-%m-%d %H:%M:%S')"
