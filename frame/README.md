# Rashomon Sets of Falling Rule Lists

This code contains algorithms to construct the Rashomon set of Falling Rule Lists (FRLs). This code is partly a reimplementation of "An Optimization Approach to Learning Falling Rule Lists" by Chen and Rudin (AISTATS 2018) in Python 3, with extended functionality to generate partially-complete Rashomon sets of Falling Rule Lists, and also some performance improvements.

A camera ready version of the codebase will be more polished. This is intended to give a sense of the experimental setup, reproducibility, and implementation details associated with our algorithms.

## Example Usage: FRAME

Here's a minimal example of how to generate a Rashomon set of Falling Rule Lists from a dataset using FRAME:

```python
import pandas as pd
from rashomon_sets import FRLRashomonSet

# Load dataset. This should be binarized. 
df = pd.read_csv('data/Australian Credit.csv')
X = df.iloc[:, :-1].astype(bool)
y = df.iloc[:, -1]

# Initialize and fit Rashomon set generator
rset = FRLRashomonSet()
rset.fit(X, y, curiosity_func="ucb+", verbose=True)

# Inspect results
print(len(rset.rset))                     # Number of unique FRL models found
print(rset.rset[0].rule_list)             # First model's rule list

# Evaluate objectives
ref_obj = rset.reference_model.objective(X, y)
rset_objs = [frl.objective(X, y) for frl in rset.rset]

print(ref_obj)                            # Objective of reference model
print(min(rset_objs))                     # Best model in Rashomon set
print(max(rset_objs))                     # Worst model in Rashomon set
```

## Example Usage: FRLFARMS

```python
import pandas as pd
from FRLFarms import FRLFarms

# Load your dataset
df = pd.read_csv("data/bank.csv")
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

frl_finder = FRLFarms(epsilon=0.02, regularization=0.01, max_sample_limit=10000)
# max_sample_limit is the number of trees FRLFarms will randomly sample if the Rashomon set of decison trees becomes too large
frl_finder.fit(X, y)

frls, scores = frl_finder.get_frls()
for rules, score in zip(frls, scores):
    print(f"FRL: {rules}, Objective: {score:.4f}")
```
There are also a number of experiments performed to test the performance and scalability of these algorithms. Some experiments are in Jupyter notebooks and can be directly run as is.
These can be found in the experiments folder.

## Key Experiments and Visualizations

Here are some scripts and notebooks used for experiments and visualizations in this project:

1. **Visualizing the Rashomon Set with TimberTrek**  
   `experiments/visualize_FRL_timbertrek.py`  
   Visualizes the Rashomon set of Falling Rule Lists (FRLs) using an interactive wheel interface.

2. **FRLFarms Runtime and Rashomon Set Size Experiments**  
      `experiments/` Contains appropriately named scripts to evaluate the runtime and Rashomon set size of `FRLFarms` under varying values of $\epsilon$ and number of antecedents.

3. **Positive Class Weight Ablation Study**  
   `FRAME_w_ablation.ipynb`  
   Jupyter notebook to analyze how the size of the Rashomon set changes as the positive class weight is varied.

4. **Regularization Parameter ($\lambda$) Ablation Study**  
   `FRAME_lambda_ablation.ipynb`  
   Jupyter notebook to analyze the impact of $\lambda$ on the Rashomon set size.

5. **Sparsity vs. Accuracy Tradeoff Visualization**  
   `FRAME_sparsity.ipynb`  
   Jupyter notebook to visualize the tradeoff between model sparsity and training accuracy across the Rashomon set of FRLs.

5. **Sparsity vs. Accuracy Tradeoff Visualization**  
   `experiments/FRAME_sparsity.ipynb`  
   Jupyter notebook to visualize the tradeoff between model sparsity and training accuracy across the Rashomon set of FRLs.

6. **Curiosity Function Comparisons and Min Support Effects on Rashomon Set Size**  
   `experiments/final_figures.ipynb`  
   Jupyter notebook to find Rashomon sets using different curiosity functions and plot their model discovery rates. Also compares the size of the rashomon set as the number of antecedents mined by FPGrowth decreases.

6. **PaCMAP Embedding of FRL Rashomon Sets**  
   `experiments/pacmap_figures.ipynb`  
   Jupyter notebook to embed FRL Rashomon sets in 2D using various distance metrics.

7. **Curiosity Function Hyperparameter Tuning**  
   `experiments/curiosity_params.ipynb`  
   Jupyter notebook to find values of $\gamma$, $\lambda_r$ and $\lambda_t$ to use with $C_{Chen}$, $C_{UCB}$ and $C_{UCB+}$ on various datasets.

8. **Model Class Reliance for FRLs**  
   `experiments/mcr_fig.ipynb`  
   Jupyter notebook to find the model class reliance of features in various datasets.

The `experiments` folder contains a number of other labelled ipynb and sh scripts used to generate different figures in the paper. 

## Dependencies (all pip installable, latest versions)
pip install numpy pandas tqdm treefarms gosdt matplotlib mlxtend gmpy2 pickle os pacmap
