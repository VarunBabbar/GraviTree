#!/bin/bash
#SBATCH --job-name=experiment_15
#SBATCH --output=experiment_15_slurms/%A_%a.out
#SBATCH --error=experiment_15_slurms/%A_%a.err
#SBATCH --array=0
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1

source ~/.bashrc
conda activate FRL

# Define the datasets
#datasets=("Australian Credit.csv" "bcw bin.csv" "Broward Data.csv" "NIJ Recidivism.csv" "bank_binary.csv" "compas.csv" "heloc_binary.csv" "spambase_binary.csv")
datasets=("spambase_binary.csv")
# Get the dataset for this array job
dataset=${datasets[$SLURM_ARRAY_TASK_ID]}

echo "Processing dataset: $dataset"

# Run the Python code for this dataset
python3 << EOF
import os
import pickle
import pandas as pd
import numpy as np
from experiment_15_helpers.bootstrap_funcs import compas_bootstrap, gen_bootstrap, ds_features

# Create intermediate directory if it doesn't exist
intermediate_dir = 'experiments_data/experiment_15_intermediates'
os.makedirs(intermediate_dir, exist_ok=True)

# Current dataset to process
dataset = "$dataset"
print(f"Processing {dataset}...")

# Load the rashomon set data
if dataset in ["Australian Credit.csv", "bcw bin.csv", "Broward Data.csv", "NIJ Recidivism.csv"]:
    with open('experiments_data/experiment_14.pkl', 'rb') as file:
        dataset_rsets = pickle.load(file)
else:
    with open('experiments_data/experiment_14b.pkl', 'rb') as file:
        dataset_rsets = pickle.load(file)

bootstrap_iters = 40

# Process the dataset
df = pd.read_csv(f'data/{dataset}')
X = df.iloc[:, :-1].astype(bool)
features = X.columns
# Make X have all the negated features
for col in X.columns: 
    X['~' + col] = ~X[col]
y = df.iloc[:, -1]

# Get the rashomon set for this dataset
rset = dataset_rsets[dataset].rset

# Get the in sample loss of each model without noise
in_sample_losses = np.array([1.0 - m.score(X, y) for m in rset])  # shape = (n_models,)
if (in_sample_losses == 0).any():
    raise ValueError("Some model has zero loss; MCR ratio would divide by zero.")

# container for features of this data set
per_feature = {}

# Loop over each feature
group_features = ds_features[dataset]
for feat in group_features:
    mcr_minus = np.empty(bootstrap_iters)
    mcr_plus = np.empty(bootstrap_iters)
    # Loop over bootstrap iterations
    for b in range(bootstrap_iters):
        # use the new gen_bootstrap to perturb X for this feature
        if dataset == 'compas.csv':
            X_boot = compas_bootstrap(X, dataset, feat)
        else:
            X_boot = gen_bootstrap(X, dataset, feat)

        # evaluate all models on the perturbed data
        in_sample_losses_noisey = np.array([1.0 - m.score(X_boot, y) for m in rset])

        mr_hat = in_sample_losses_noisey / in_sample_losses
        mcr_minus[b] = mr_hat.min()
        mcr_plus[b]  = mr_hat.max()
    per_feature[feat] = {"mcr_minus": mcr_minus, "mcr_plus": mcr_plus}

# Save intermediate results for this dataset
dataset_name = os.path.splitext(dataset)[0]
intermediate_path = os.path.join(intermediate_dir, f'{dataset_name}_mcr.pkl')
with open(intermediate_path, 'wb') as file:
    pickle.dump(per_feature, file)

print(f"Successfully processed {dataset}")
EOF

# Check if the processing was successful
if [ $? -eq 0 ]; then
    echo "Successfully completed processing $dataset"
else
    echo "Failed to process $dataset"
    exit 1
fi
