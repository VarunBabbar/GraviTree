#!/bin/bash
#SBATCH --job-name=experiment_18_ucb+
#SBATCH --output=experiment_18_slurms/ucb+/%A_%a.out
#SBATCH --error=experiment_18_slurms/ucb+/%A_%a.err
#SBATCH --array=1-128
#SBATCH --mem=10G
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --partition=common

source ~/.bashrc
conda activate FRL

# Create directories for outputs
mkdir -p experiments_data/experiment18_intermediates/ucb+

# Define datasets and weights
declare -a datasets=(
    "Australian Credit.csv"
    "bcw bin.csv"
    "Broward Data.csv"
    "NIJ Recidivism.csv"
    "bank_binary.csv"
    "compas.csv"
    "heloc_binary.csv"
    "spambase_binary.csv"
)

declare -a exploration_weights=(0.1 0.5 1 2)
declare -a exploitation_weights=(0.1 0.5 1 2)

# Calculate indices
# We have 8 datasets, 4 exploration weights, and 4 exploitation weights (128 combinations)
index=$((SLURM_ARRAY_TASK_ID - 1))
dataset_index=$((index / 16))
remaining=$((index % 16))
exploration_index=$((remaining / 4))
exploitation_index=$((remaining % 4))

# Get the parameters for this task
DATASET=${datasets[$dataset_index]}
EXPLORATION=${exploration_weights[$exploration_index]}
EXPLOITATION=${exploitation_weights[$exploitation_index]}

echo "Processing dataset: $DATASET with exploration: $EXPLORATION, exploitation: $EXPLOITATION"

# Run the Python script
python - <<EOF
import pandas as pd
import pickle
import time
import os
import numpy as np
from FRL import *
from rashomon_sets import *

# Parameters
dataset = "$DATASET"
exploration_weight = $EXPLORATION
exploitation_weight = $EXPLOITATION
curiosity_function = 'ucb+'
runs = 5
epsilon = 0.01
C = 0.005
first_pass_iters = 10000
second_pass_iters = 20000

print(f"Processing dataset: {dataset} with exploration: {exploration_weight}, exploitation: {exploitation_weight}")

# Define reference objectives
ref_objs = {
    'Australian Credit.csv': 0.141875,
    'bcw bin.csv': 0.06507153075822604,
    'Broward Data.csv': 0.39154554759467763,
    'NIJ Recidivism.csv': 0.3202754549926217,
    'bank_binary.csv': 0.11227715107277152,
    'compas.csv': 0.348666481840865,
    'heloc_binary.csv': 0.3019973228798164,
    'spambase_binary.csv': 0.13673913043478259,
}

# Define intermediate file path
safe_dataset = dataset.replace(' ', '_').replace('.', '_')
intermediate_path = f'experiments_data/experiment18_intermediates/ucb+/{safe_dataset}_expl_{exploration_weight}_expt_{exploitation_weight}.pkl'

# Check if intermediate result already exists
if os.path.exists(intermediate_path):
    print(f"Intermediate result already exists. Skipping.")
    exit(0)

# Get reference objective for this dataset
ref_obj = ref_objs[dataset]

# Run multiple times and collect results
param_model_counts = []
for run in range(1, runs + 1):    
    # Load dataset
    df = pd.read_csv(f'data/{dataset}')
    X = df.iloc[:, :-1].astype(bool)
    y = df.iloc[:, -1]
    
    # Create and fit the rashomon set
    rset = FRLRashomonSet(epsilon=epsilon, C=C)
    rset.fit(X, y, verbose=False,
            first_pass_iters=first_pass_iters,
            second_pass_iters=second_pass_iters,
            curiosity_func=curiosity_function,
            best_obj=ref_obj,
            exploration_weight=exploration_weight,
            exploitation_weight=exploitation_weight)
    
    n_models = rset.unique_models
    param_model_counts.append(n_models)

# Save intermediate result
with open(intermediate_path, 'wb') as f:
    pickle.dump(param_model_counts, f)
print(f"Saved intermediate result to {intermediate_path}")

print(f"Completed processing for this parameter combination")
EOF

echo "Job completed for dataset: $DATASET with exploration: $EXPLORATION, exploitation: $EXPLOITATION"
