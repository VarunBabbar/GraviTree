
## GraviTree 
Camera ready code + experiments for the **ICML 2026 Spotlight** paper [Falling Trees: A Model Class for Interpretable Risk Prioritization](https://arxiv.org/abs/2609.23780). The experiment code is included to make replication of the results of our paper easier - it includes code for running comparisons between falling trees, regular trees, and rule lists, as well as hyperparameter sweeps of the model. 


# Algorithm

A **falling tree** is a decision tree whose leaf risks decrease as you move away from the root: along every root-to-leaf path, leaves that branch off earlier have higher positive-class probability than leaves that branch off later. High-risk cases are therefore identified with the fewest conditions. Falling rule lists are the special case with a single path.

**Objective.** GraviTree minimizes

    L(T) = misclassification rate + branching_cost × H(T)

where H(T) counts internal nodes whose children are *both* internal. `branching_cost` controls the model's shape: a high value forces a falling rule list, a low value allows a full tree, and values in between give trees that are almost rule lists (see the figures below). `depth` bounds tree depth, and `min_support` requires every leaf to cover at least that fraction of the data. The falling constraint is enforced as a hard constraint.

**Search.** GraviTree uses dynamic programming with bounds over subproblems defined by (data subset, remaining depth, risk ceiling). The risk ceiling is the probability of the most recent leaf adjacent to the current path. A node's leaf probability must be at least the maximum leaf probability in its sibling subtree, which gives a local check for the global constraint. This lets GraviTree:
- **prune splits early:** any split whose child exceeds the inherited ceiling can never yield a falling tree.
- **couple siblings:** the higher-risk child is solved first, and its leaf probability becomes a tighter ceiling for the other child.
- **bound and cache:** each recursive call gets a loss budget and stops as soon as lower bounds exceed it. Solutions are cached by (subset, depth, quantized ceiling).

**Rashomon set.** Rather than a single tree, GraviTree returns every falling tree with loss ≤ (1 + `eps`) × optimal loss. It builds this set by recursively enumerating near-optimal subtrees for each child, then merging the sorted child sets under the remaining budget. Pairs that violate the falling constraint are dropped. The set is stored in `model.ft_rset`, sorted by loss, so `tree_idx=0` is the optimal tree.

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


