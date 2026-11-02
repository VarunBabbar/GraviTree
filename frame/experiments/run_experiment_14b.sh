#!/bin/bash
#SBATCH --job-name=experiment_14b
#SBATCH --output=experiment_14b_slurms/slurm_%A_%a.out
#SBATCH --error=experiment_14b_slurms/slurm_%A_%a.err
#SBATCH --array=1-4
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --partition=common

# Activate your conda environment if needed
source ~/.bashrc
conda activate FRL  # Replace with your actual environment name

# Create directories for outputs
mkdir -p experiment_14b_slurms
mkdir -p experiments_data/experiment14b_intermediates

# Define datasets
declare -a datasets=("bank_binary.csv" "compas.csv" "heloc_binary.csv" "spambase_binary.csv")

# Get the dataset for this array task
DATASET=${datasets[$((SLURM_ARRAY_TASK_ID-1))]}

echo "Processing dataset: $DATASET"

# Run the Python script
python - <<EOF
import pandas as pd
import pickle
import time
import os
from rashomon_sets import *
from FRL import *

# Parameters
dataset = "$DATASET"
epsilon = 0.01
C = 0.005
first_pass_iters = 100000
second_pass_iters = 1000000
curiosity_func = 'ucb+'

match dataset:
    case 'bank_binary.csv':
        curiosity_func = 'uniform'
    case 'compas.csv':
        epsilon = 0.007
    case 'heloc_binary.csv':
        second_pass_iters = 500000
    case 'spambase_binary.csv':
        epsilon = 0.015
        second_pass_iters = 500000

print(f"Processing dataset: {dataset}")

# Define intermediate file path
intermediate_dir = 'experiments_data/experiment14b_intermediates'
intermediate_path = os.path.join(intermediate_dir, f'{dataset}_rset.pkl')

# Check if intermediate result already exists
if os.path.exists(intermediate_path):
    print(f"Intermediate result for {dataset} already exists. Loading from file.")
    with open(intermediate_path, 'rb') as f:
        rset = pickle.load(f)
else:
    print(f"Computing rashomon set for {dataset}...")
    # Load dataset
    df = pd.read_csv(f'data/{dataset}')
    X = df.iloc[:, :-1].astype(bool)
    y = df.iloc[:, -1]
    
    # Create and fit the rashomon set
    start_time = time.perf_counter()
    rset = FRLRashomonSet(epsilon=epsilon, C=C)
    rset.fit(X, y, verbose=False,
            first_pass_iters=first_pass_iters,
            second_pass_iters=second_pass_iters,
            curiosity_func=curiosity_func)
    end_time = time.perf_counter()
    
    print(f"Computation time: {end_time - start_time:.2f} seconds")
    
    # Save intermediate result
    with open(intermediate_path, 'wb') as f:
        pickle.dump(rset, f)
    print(f"Saved intermediate result to {intermediate_path}")

print(f"Completed processing for {dataset}")
EOF

echo "Job completed for dataset: $DATASET"
