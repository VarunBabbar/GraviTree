
## GraviTree 
Camera ready code + experiments for the **ICML 2026 Spotlight** paper [Falling Trees: A Model Class for Interpretable Risk Prioritization](https://arxiv.org/abs/2609.23780). The experiment code is included to make replication of the results of our paper easier - it includes code for running comparisons between falling trees, regular trees, and rule lists, as well as hyperparameter sweeps of the model. 

# Algorithm

A **falling tree** is a decision tree in which higher-risk leaves sit closer to the root. Specifically, a leaf that branches off a path must have a risk at least as high as every leaf below it on that side. As a result, the highest-risk cases are identified with the fewest conditions. Falling rule lists are the special case with a single path.

**Example** (illustrative):

    priors > 3?
    ├── yes → risk 0.71
    └── no: age < 25?
        ├── yes: juvenile crimes > 0?
        │   ├── yes → risk 0.58
        │   └── no  → risk 0.40
        └── no: male?
            ├── yes → risk 0.33
            └── no  → risk 0.19

The leaf at 0.71 is valid because it is at least the maximum risk in its sibling subtree (0.58). If it were 0.50, the tree would not be falling.

**Objective.** GraviTree minimizes misclassification + `branching_cost` × (number of nodes whose two children are both internal). A high `branching_cost` gives a rule list and a low one gives a full tree. `depth` bounds tree depth, and `min_support` sets the minimum fraction of data each leaf must cover.

**Search.** GraviTree uses dynamic programming with bounds. Each subproblem carries a risk ceiling inherited from its ancestors. Splits that exceed the ceiling are pruned, and subproblems are cached by (subset, depth, quantized ceiling). It returns the **Rashomon set**: all falling trees with loss ≤ (1 + `eps`) × the optimal loss, stored in `model.ft_rset` and sorted by loss (`tree_idx=0` is the optimum).


# Installation

Requires Python 3.10 or newer and a C++17 compiler (`g++`). From an environment that is already active, install this repository. That installs the Python dependencies and compiles the GraviTree source code. 

```bash
python -m pip install .
```

`uv pip install .` does the same thing into the active environment. `pybind11` is a build dependency, so the installer provides it while compiling; you do not install it separately.

# Example Usage

Run this from the repository root. `binarize=True` turns continuous columns into threshold features. `score` and `print_tree` use tree `i` from the Rashomon set. `example.ipynb` also runs FICO and scores every tree.

```python
import pandas as pd
from falling_trees.gravitree import GraviTree

df = pd.read_csv("data/benchmark/compas.csv")
X = df.drop(columns=["two_year_recid"])
y = df["two_year_recid"].astype(int)

model = GraviTree(depth=5, eps=0.02, min_support=0.05, branching_cost=0.05, binarize=True)
model.fit(X, y)
i = 0  # index into model.ft_rset; 0 is the minimum-loss tree
accuracy = model.score(X, y, tree_idx=i)
model.print_tree(tree_idx=i)
```


### Falling tree with high branching cost (rule list)
<img width="565" height="436" alt="Screenshot 2026-11-20 at 3 39 24 PM" src="https://github.com/user-attachments/assets/93fe60a0-7d07-449b-83bf-bf52920f2c76" />

### Falling Tree with medium branching cost (almost rule list)
<img width="703" height="442" alt="Screenshot 2026-11-20 at 3 39 39 PM" src="https://github.com/user-attachments/assets/f79ae8be-cf6a-4908-8130-fce679ac8767" />

### Falling Tree with low branching cost (close to a full tree)
<img width="725" height="365" alt="Screenshot 2026-11-20 at 3 39 55 PM" src="https://github.com/user-attachments/assets/f36ee0e2-848b-4556-837c-090234c397cc" />


