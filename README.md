## GraviTree 
Camera Ready code for the **ICML 2026 Spotlight** paper [Falling Trees: A Model Class for Interpretable Risk Prioritization](https://arxiv.org/abs/2609.23780).

# Installation

Requires Python 3.10 or newer and a C++17 compiler (`g++`). From an environment that is already active, install this repository. That installs the Python dependencies and compiles the GraviTree source code. 

```bash
pip install .
```

`uv pip install .` does the same thing into the active environment. `pybind11` is a build dependency, so the installer provides it while compiling; you do not install it separately.

# Example Usage

Run this from the repository root. `binarize=True` turns continuous columns into threshold features. `score` and `print_tree` use tree `i` from the Rashomon set. `example.ipynb` also runs FICO and scores every tree.

```python
import pandas as pd
from falling_trees.gravitree import GraviTree

df = pd.read_csv("path_to_gravitree_/../../data/benchmark/compas.csv")
X = df.drop(columns=["two_year_recid"])
y = df["two_year_recid"].astype(int)

model = GraviTree(depth=5, eps=0.02, min_support=0.05, branching_cost=0.05, binarize=True)
model.fit(X, y)
i = 0  # index into model.ft_rset; 0 is the minimum-loss tree
accuracy = model.score(X, y, tree_idx=i)
model.print_tree(tree_idx=i)
```
