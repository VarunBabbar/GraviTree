#!/bin/bash
#SBATCH --job-name=experiment_9_dist
#SBATCH --output=experiment_9_slurms/%A_%a.out
#SBATCH --error=experiment_9_slurms/%A_%a.err
#SBATCH --array=0-7
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1

source ~/.bashrc
conda activate FRL  # Replace with your actual environment name

# Run the Python script with the dataset index as an argument
python - <<EOF
import pickle
import pandas as pd
import numpy as np
from rashomon_sets import *
from FRL import *
from rset_analysis import *

# Define datasets
datasets = ['bcw', 'Broward', 'NIJ', 'Australian Credit', 'bank_binary', 'compas', 'heloc_binary', 'spambase_binary']
dataset_file_names = {'bcw': 'data/bcw bin.csv',
                      'Broward': 'data/Broward Data.csv',
                      'NIJ': 'data/NIJ Recidivism.csv',
                      'Australian Credit': 'data/Australian Credit.csv',
                      'bank_binary': 'data/bank_binary.csv',
                      'compas': 'data/compas.csv',
                      'heloc_binary': 'data/heloc_binary.csv',
                      'spambase_binary': 'data/spambase_binary.csv'}
distance_metrics = ['tree_edit', 'prediction', 'feature_set']

# Get the dataset index from SLURM_ARRAY_TASK_ID
dataset_idx = int("${SLURM_ARRAY_TASK_ID}")
dataset = datasets[dataset_idx]

print(f"Processing dataset: {dataset}")

# Load the rset
rset_path = f'experiments_data/experiment9_intermediates/{dataset}_rset.pkl'
with open(rset_path, 'rb') as file:
    rset = pickle.load(file)

# Load the dataset
df = pd.read_csv(dataset_file_names[dataset])
X = df.iloc[:, :-1].astype(bool)
for col in X.columns:
    X['~' + col] = ~X[col]
y = df.iloc[:, -1]

model_feature_lookup_list = list(rset.reference_model.features)
if len(rset.rset) > 1500:
    rset.rset = rset.rset[:1500]

print(f"Calculating distance matrices for {dataset}...")

# Calculate distance matrices
feature_set_distance_matrix = calc_distance_matrix(rset.rset, distance_metric='feature_set_hamming', model_feature_lookup_list=model_feature_lookup_list)
edit_distance_matrix = calc_distance_matrix(rset.rset, distance_metric='levenshtein')
pred_distance_matrix = calc_distance_matrix(rset.rset, distance_metric='prediction', X=X)

distance_matrices = {
    'feature_set': feature_set_distance_matrix,
    'tree_edit': edit_distance_matrix,
    'prediction': pred_distance_matrix
}

# Save the distance matrices
distance_matrix_path = f'experiments_data/experiment9_intermediates/{dataset}_distance_matrices.pkl'
with open(distance_matrix_path, 'wb') as file:
    pickle.dump(distance_matrices, file)

print(f"Completed processing dataset: {dataset}")
EOF
