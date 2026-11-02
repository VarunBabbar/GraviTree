#!/bin/bash
#SBATCH --job-name=ft_vs_regular
#SBATCH --time=48:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-145
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
NUM_TRIALS=1
NUM_ESTIMATORS=100

# Branching costs to sweep over
BRANCHING_COSTS=(0.0 0.005 0.01 0.02 0.05 0.1)

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
DATASET_PATH="${DATA_DIR}/${DATASET}.csv"

echo "=========================================="
echo "SLURM JOB ID:       $SLURM_JOB_ID"
echo "ARRAY TASK ID:      $SLURM_ARRAY_TASK_ID"
echo "DATASET:            $DATASET"
echo "BRANCHING_COST:     $BRANCHING_COST"
echo "NUM_TRIALS:         $NUM_TRIALS"
echo "DEPTH:              $DEPTH"
echo "LAM:                $LAM"
echo "EPS:                $EPS"
echo "MIN_SUPPORT:        $MIN_SUPPORT"
echo "NUM_ESTIMATORS:     $NUM_ESTIMATORS"
echo "=========================================="

# ========================
# Run experiment
# ========================

export PYTHONUNBUFFERED=1

python3 falling_trees_vs_regular_trees.py \
  --dataset "$DATASET_PATH" \
  --branching-costs "$BRANCHING_COST" \
  --depth "$DEPTH" \
  --lam "$LAM" \
  --eps "$EPS" \
  --min-support "$MIN_SUPPORT" \
  --num-trials "$NUM_TRIALS" \
  --num-estimators "$NUM_ESTIMATORS" \
  --output-dir "falling_trees_vs_regular_trees_results"

STATUS=$?

echo ""
if [ $STATUS -eq 0 ]; then
  echo "SUCCESS: dataset=$DATASET, branching_cost=$BRANCHING_COST"
else
  echo "FAILED with status $STATUS: dataset=$DATASET, branching_cost=$BRANCHING_COST"
fi

echo "=========================================="

