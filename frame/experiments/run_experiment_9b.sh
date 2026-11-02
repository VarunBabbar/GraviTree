#!/bin/bash
#SBATCH --job-name=experiment_9b
#SBATCH --output=experiment_9b_slurms/%A_%a.out
#SBATCH --error=experiment_9b_slurms/%A_%a.err
#SBATCH --array=0-3
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=1

source ~/.bashrc
conda activate FRL

# Create output directory for Slurm logs
mkdir -p experiment_9b_slurms

# Define the datasets
datasets=("bank_binary.csv" "compas.csv" "heloc_binary.csv" "spambase_binary.csv")

# Get the dataset for this array job
dataset=${datasets[$SLURM_ARRAY_TASK_ID]}

echo "Processing dataset: $dataset"

# Run the Python code for this dataset
python3 << EOF
import pandas as pd
import pickle
import time
import os
from rashomon_sets import *
from FRL import *

# Get dataset from environment variable
dataset = "$dataset"

# Parameters
epsilon = 0.01
C = 0.005
first_pass_iters = 50000
second_pass_iters = 500000
curiosity_func = 'ucb+'

if dataset == 'bank_binary.csv':
    curiosity_func = 'paper'

# Process the dataset
try:
    df = pd.read_csv(f'data/{dataset}')
    X = df.iloc[:, :-1].astype(bool)
    y = df.iloc[:, -1]

    rset = FRLRashomonSet(epsilon=epsilon, C=C)
    rset.fit(X, y, verbose=False,
            first_pass_iters=first_pass_iters,
            second_pass_iters=second_pass_iters,
            curiosity_func=curiosity_func)

    # Ensure intermediate directory exists
    intermediate_dir = 'experiments_data/experiment9_intermediates'
    # Remove .csv extension for the output file name
    dataset_name = os.path.splitext(dataset)[0]
    intermediate_path = os.path.join(intermediate_dir, f'{dataset_name}_rset.pkl')
    os.makedirs(intermediate_dir, exist_ok=True)
    
    # Save intermediate results for this dataset
    with open(intermediate_path, 'wb') as file:
        pickle.dump(rset, file)
    
    print(f"Successfully processed {dataset}")
    
except Exception as e:
    print(f"Error processing {dataset}: {str(e)}")
    exit(1)
EOF

# Check if the processing was successful
if [ $? -eq 0 ]; then
    echo "Successfully completed processing $dataset"
else
    echo "Failed to process $dataset"
    exit 1
fi
