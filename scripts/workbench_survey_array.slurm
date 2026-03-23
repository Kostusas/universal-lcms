#!/bin/bash
#SBATCH --job-name=workbench-survey
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/%x_%A_%a.out
#SBATCH --error=slurm/%x_%A_%a.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
MANIFEST_PATH="${MANIFEST_PATH:-$REPO_ROOT/notes/metabolomics_workbench_candidate_manifest.json}"
CANDIDATE_GROUP="${CANDIDATE_GROUP:-candidates}"
CANDIDATE_INDEX="${CANDIDATE_INDEX:-${SLURM_ARRAY_TASK_ID:-}}"

if [[ -z "$CANDIDATE_INDEX" ]]; then
  echo "CANDIDATE_INDEX is not set and SLURM_ARRAY_TASK_ID is empty." >&2
  exit 2
fi

DOWNLOAD_DIR_DEFAULT="/tmp/universal-lcms-workbench-downloads-${USER:-user}"

WORKBENCH_DOWNLOAD_DIR="${WORKBENCH_DOWNLOAD_DIR:-$DOWNLOAD_DIR_DEFAULT}"
WORKBENCH_OUTPUT_DIR="${WORKBENCH_OUTPUT_DIR:-$HOME/universal-lcms-workbench-runs}"

mkdir -p "$REPO_ROOT/slurm" "$WORKBENCH_DOWNLOAD_DIR" "$WORKBENCH_OUTPUT_DIR"
cd "$REPO_ROOT"

echo "[$(date -Iseconds)] repo=$REPO_ROOT group=$CANDIDATE_GROUP index=$CANDIDATE_INDEX"

PYTHONPATH=src pixi run python "$REPO_ROOT/scripts/run_workbench_candidate.py" \
  --manifest "$MANIFEST_PATH" \
  --group "$CANDIDATE_GROUP" \
  --index "$CANDIDATE_INDEX" \
  --download-dir "$WORKBENCH_DOWNLOAD_DIR" \
  --output-dir "$WORKBENCH_OUTPUT_DIR" \
  "$@"
