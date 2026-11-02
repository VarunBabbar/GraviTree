#!/bin/bash
#SBATCH --job-name=bc_sweep
#SBATCH --time=02:00:00
#SBATCH --partition=compsci
#SBATCH --mem=200G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-134
#SBATCH --output=logs/%x_%A_%a.log

# ========================
# Setup
# ========================

mkdir -p logs

# Fixed hyperparameters
DEPTH=5
LAM=0.005
EPS=0.02
MIN_SUPPORT=0.02

# Branching costs to sweep over
BRANCHING_COSTS=(0.0 0.005 0.01 0.02 0.03 0.04 0.05 0.075 0.1)

# Get all datasets from benchmark directory
DATA_DIR="../data/benchmark"
DATASETS=()
for csv_file in ${DATA_DIR}/*.csv; do
    if [ -f "$csv_file" ]; then
        dataset_name=$(basename "$csv_file" .csv)
        DATASETS+=("$dataset_name")
    fi
done

# ========================
# Build configuration list
# ========================

CONFIGS=()

for dataset in "${DATASETS[@]}"; do
  for branching_cost in "${BRANCHING_COSTS[@]}"; do
    CONFIGS+=("$dataset|$branching_cost")
  done
done

TOTAL_CONFIGS=${#CONFIGS[@]}

# Validate array task ID
if [ "$SLURM_ARRAY_TASK_ID" -ge "$TOTAL_CONFIGS" ]; then
  echo "Invalid task ID $SLURM_ARRAY_TASK_ID (max: $((TOTAL_CONFIGS - 1)))"
  exit 1
fi

# ========================
# Decode this task's config
# ========================

IFS="|" read -r DATASET BRANCHING_COST <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

echo "=========================================="
echo "SLURM JOB ID:       $SLURM_JOB_ID"
echo "ARRAY TASK ID:      $SLURM_ARRAY_TASK_ID"
echo "DATASET:            $DATASET"
echo "BRANCHING_COST:     $BRANCHING_COST"
echo "DEPTH:              $DEPTH"
echo "LAM:                $LAM"
echo "EPS:                $EPS"
echo "MIN_SUPPORT:        $MIN_SUPPORT"
echo "=========================================="

# ========================
# Run experiment
# ========================

python3 branching_cost_sweep.py \
  --dataset "$DATASET" \
  --branching_cost "$BRANCHING_COST" \
  --depth "$DEPTH" \
  --lam "$LAM" \
  --eps "$EPS" \
  --min_support "$MIN_SUPPORT"

STATUS=$?

echo ""
if [ $STATUS -eq 0 ]; then
  echo "SUCCESS: dataset=$DATASET, branching_cost=$BRANCHING_COST"
else
  echo "FAILED with status $STATUS: dataset=$DATASET, branching_cost=$BRANCHING_COST"
fi

echo "=========================================="

