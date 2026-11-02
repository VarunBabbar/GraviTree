from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Iterable, Set
import math
import copy
import numpy as np
from tqdm import tqdm
from collections import Counter
import heapq
import sys
sys.path.append('/usr/project/xtmp/vb97/FRL_Rashomon_Set/falling-models')
from falling_trees.binarize_dataset import binarize_dataset
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from collections import defaultdict, OrderedDict
import pandas as pd
import pstats
import logging
import io
import bisect
import os
import pickle as pkl
import time


class TrieNode:

    def __init__(self,
                 feature: int,
                 left: Trie,
                 right: Trie,
                 min_objective: float,
                 is_leaf=False,
                 pred_prob: float = 0.5):
        self.feature = feature
        self.left = left
        self.right = right
        self.min_objective = min_objective
        self.is_leaf = is_leaf  # Use the parameter value, not hardcoded False
        self.pred_prob = pred_prob  # Store prediction probability for leaves
        self.list_of_objectives = []  # Initialize as empty list
        heapq.heappush(self.list_of_objectives, min_objective)


class Trie:

    def __init__(self, budget: float):
        self.budget = budget
        self.children: list[TrieNode] = []  # Heap for getting best tree
        heapq.heapify(self.children)
        self.children_by_feature: Dict[int, TrieNode] = {}  # O(1) lookup by feature
        self.leaf_node: Optional[TrieNode] = None  # Cache leaf node if exists
        self._tie_breaker = 0  # Counter to break ties when objectives are equal
        self._solution_cache: Dict[frozenset, Optional[Node | Leaf]] = {}  # Cache for return_solution_to_subproblem

    def add_new_tree_to_trie(self, tree: Node | Leaf):
        if isinstance(tree, Leaf):
            # Only add leaf if we don't have one or this one is better
            if self.leaf_node is None or tree.objective < self.leaf_node.min_objective:
                if self.leaf_node is not None:
                    # Remove old leaf from heap (lazy removal - will be filtered in get_best_tree)
                    pass
                self.leaf_node = TrieNode(None, None, None, tree.objective, is_leaf=True, pred_prob=tree.pred_prob)
                heapq.heappush(self.children, (tree.objective, self._tie_breaker, self.leaf_node))
                self._tie_breaker += 1
        elif isinstance(tree, Node):
            # O(1) lookup instead of O(n) linear search
            if tree.feature in self.children_by_feature:
                child_node = self.children_by_feature[tree.feature]
                if tree.objective < child_node.min_objective:
                    child_node.min_objective = tree.objective
                    heapq.heappush(child_node.list_of_objectives, tree.objective)
                # Recursively add subtrees
                child_node.left.add_new_tree_to_trie(tree.left)
                child_node.right.add_new_tree_to_trie(tree.right)
            else:
                # New feature - create new TrieNode
                new_trienode = TrieNode(tree.feature,
                                        left=Trie(budget=self.budget),
                                        right=Trie(budget=self.budget),
                                        min_objective=tree.objective)
                new_trienode.left.add_new_tree_to_trie(tree.left)
                new_trienode.right.add_new_tree_to_trie(tree.right)
                self.children_by_feature[tree.feature] = new_trienode
                heapq.heappush(self.children, (tree.objective, self._tie_breaker, new_trienode))
                self._tie_breaker += 1

    def get_best_tree_from_trie(self) -> Node | Leaf:
        if len(self.children) == 0:
            return None

        best_trie_node = self.children[0][2]  # Updated index from [1] to [2] due to tie_breaker

        if best_trie_node.is_leaf:
            # Convert TrieNode to Leaf object
            return Leaf(pred_prob=best_trie_node.pred_prob, objective=best_trie_node.min_objective)
        else:
            # Recursively get best trees from left and right subtrees
            # Handle case where subtrees might be empty (return None)
            left_tree = best_trie_node.left.get_best_tree_from_trie() if best_trie_node.left is not None else None
            right_tree = best_trie_node.right.get_best_tree_from_trie() if best_trie_node.right is not None else None

            # If either subtree is None, we can't form a complete tree
            if left_tree is None or right_tree is None:
                return None

            node = Node(feature=best_trie_node.feature,
                        left=left_tree,
                        right=right_tree,
                        objective=best_trie_node.min_objective)
            return node

    def count_all_trees(self, budget: float = None) -> int:
        count = 0

        # Count leaf node if exists
        if self.leaf_node is not None:
            if budget is None or self.leaf_node.min_objective <= budget:
                count += 1

        # Count nodes by feature (more efficient than iterating heap)
        for trie_node in self.children_by_feature.values():
            if budget is None:
                count += len(trie_node.list_of_objectives) if trie_node.list_of_objectives else 1
            else:
                for obj in trie_node.list_of_objectives:
                    if obj <= budget:
                        count += 1

        return count

    def return_solution_to_subproblem(self, subproblem_path: frozenset):
        # Check cache first
        if subproblem_path in self._solution_cache:
            return self._solution_cache[subproblem_path]

        result = None
        if not subproblem_path:
            result = self.get_best_tree_from_trie()
        else:
            # Try each feature in the path to find a valid traversal
            # Since order doesn't matter for equivalent paths, we try all features
            for feature, direction in subproblem_path:
                # O(1) lookup instead of O(n) linear search
                if feature in self.children_by_feature:
                    child_node = self.children_by_feature[feature]
                    # Build remaining path by removing this feature:direction pair
                    remaining_path = subproblem_path - frozenset({(feature, direction)})
                    if direction == '0':
                        if child_node.left is not None:
                            candidate = child_node.left.return_solution_to_subproblem(remaining_path)
                            if candidate is not None:
                                result = candidate
                                break  # Found a valid solution, no need to try other features
                    else:
                        if child_node.right is not None:
                            candidate = child_node.right.return_solution_to_subproblem(remaining_path)
                            if candidate is not None:
                                result = candidate
                                break  # Found a valid solution, no need to try other features

        # Cache the result (even if None) to avoid recomputation
        self._solution_cache[subproblem_path] = result
        return result


@dataclass(frozen=True)
class Leaf:
    pred_prob: float  # proportion of positives in leaf
    objective: float  # objective value for this leaf (e.g., negative log-likelihood + reg)

    def is_leaf(self) -> bool:
        return True


@dataclass(frozen=True)
class Node:
    feature: int
    left: Any  # Node or Leaf
    right: Any  # Node or Leaf
    objective: float  # error / n for the node rooted at this tree. n is a global constant.

    def is_leaf(self) -> bool:
        return False


# --- Utilities ---
def pos_prop(X: np.ndarray, y: np.ndarray, row_idx: np.ndarray) -> float:
    """Return proportion of positive labels in dataset."""
    if len(row_idx) == 0:
        return 0.0
    pos = y[row_idx].sum()
    return pos / len(row_idx)


def leaf_cost(X: np.ndarray, y: np.ndarray, row_idx: np.ndarray, lam: float = 0.0, n=None) -> Tuple[float, float]:
    """Compute leaf loss: binary cross-entropy (negative log-likelihood) with Laplace smoothing to avoid log(0).
    Returns (loss, pred_prob)
    """
    if len(row_idx) == 0:
        # empty leaf: conventionally use loss 0 and prob 0.0 (or high penalty). We pick 0 loss but that generally won't be chosen.
        return 0.0, 0.0
    pos = y[row_idx].sum()
    # Laplace smoothing with +1/+2
    # p = (pos + 1) / (n + 2)
    neg = len(row_idx) - pos
    # simple 0-1 loss
    loss = min(pos / n, neg / n)
    # add simple regularization term (lam * number of samples) if desired (kept small by default)
    loss += lam
    return loss, pos / len(row_idx)


def tree_obj(tree: Any) -> float:
    """Compute total objective (sum of leaf losses) for tree."""
    if tree is None:
        return 0.0
    if isinstance(tree, Leaf):
        return tree.objective
    return tree_obj(tree.left) + tree_obj(tree.right)


def get_depth_of_pmax(tree: Any, current_depth: int) -> int:
    """
    Find the depth (relative to current_depth) where the maximum probability occurs.
    Returns the depth value (larger = shallower, smaller = deeper).
    """
    if isinstance(tree, Leaf):
        return current_depth

    left_pmax = get_max_prob(tree.left)
    right_pmax = get_max_prob(tree.right)

    if left_pmax > right_pmax:
        return get_depth_of_pmax(tree.left, current_depth - 1)
    elif right_pmax > left_pmax:
        return get_depth_of_pmax(tree.right, current_depth - 1)
    else:
        # Equal pmax, return the deeper one (smaller depth value)
        left_depth = get_depth_of_pmax(tree.left, current_depth - 1)
        right_depth = get_depth_of_pmax(tree.right, current_depth - 1)
        return min(left_depth, right_depth)


def cache_subtree_solutions(tree: Any,
                            path: frozenset,
                            depth: int,
                            n: int):
    """
    Recursively cache optimal solutions for all subtrees in the optimal tree.
    This allows OptFallingRset to reuse optimal subproblem solutions.
    
    Args:
        tree: The tree (Node or Leaf) to cache
        path: The path (frozenset of (feature, direction) tuples) to this subtree
        depth: The depth of this subtree (remaining depth from root)
        n: Total dataset size (for normalization)
    """
    if tree is None:
        return

    if isinstance(tree, Leaf):
        # Cache leaf solution
        obj = tree.objective
        pmax = tree.pred_prob
        depth_of_pmax = depth
        is_internal = False
        cached_value = _get_subproblem_optimal_objective_value((path, depth))
        if cached_value is None or obj < cached_value[0]:
            _set_subproblem_optimal_objective((path, depth), (obj, pmax, depth_of_pmax, is_internal))
        return

    # Cache internal node solution
    obj = tree.objective
    pmax = get_max_prob(tree)
    depth_of_pmax = get_depth_of_pmax(tree, depth)
    is_internal = True

    cached_value = _get_subproblem_optimal_objective_value((path, depth))
    if cached_value is None or obj < cached_value[0]:
        _set_subproblem_optimal_objective((path, depth), (obj, pmax, depth_of_pmax, is_internal))

    # Recursively cache left and right subtrees
    if isinstance(tree, Node):
        new_path_left = path | frozenset({(tree.feature, '0')})
        new_path_right = path | frozenset({(tree.feature, '1')})
        cache_subtree_solutions(tree.left, new_path_left, depth - 1, n)
        cache_subtree_solutions(tree.right, new_path_right, depth - 1, n)


def max_leaf_prob(tree: Any) -> float:
    """Return maximum positive-probability among leaves of the tree."""
    if isinstance(tree, Leaf):
        return tree.pred_prob
    return max(max_leaf_prob(tree.left), max_leaf_prob(tree.right))


def get_max_prob(tree: Any) -> float:
    """Helper to find the maximum probability in a subtree."""
    if isinstance(tree, Leaf):
        return tree.pred_prob
    return max(get_max_prob(tree.left), get_max_prob(tree.right))


def get_max_prob_at_every_level(tree: Any) -> Dict[int, float]:
    """Helper to find the maximum probability in a subtree at every level."""
    if isinstance(tree, Leaf):
        return {0: tree.pred_prob}
    left_max_prob = get_max_prob_at_every_level(tree.left)
    right_max_prob = get_max_prob_at_every_level(tree.right)
    new_dict = {}
    for l in range(0, max(left_max_prob.keys(), right_max_prob.keys()) + 1):
        new_dict[l] = max(left_max_prob.get(l, 0), right_max_prob.get(l, 0))
    return new_dict


def is_falling_tree(tree: Any, verbose: bool = False) -> bool:
    """
    Checks if a tree satisfies the 'Falling' property defined in the algorithm.
    
    Definition derived from your code:
    1. If Left child is Leaf: Left.prob >= Max(Right Subtree)
    2. If Right child is Leaf: Right.prob <= Max(Left Subtree)
    3. Recursively holds for all sub-nodes.
    """
    # Base case: A single leaf is always valid
    if isinstance(tree, Leaf):
        return True

    if not isinstance(tree.left, Leaf) and not isinstance(tree.right, Leaf):
        return is_falling_tree(tree.left, verbose) and is_falling_tree(tree.right, verbose)

    # Check Right Child Constraint
    if isinstance(tree.right, Leaf):
        if not isinstance(tree.left, Leaf):
            max_left_prob = get_max_prob(tree.left)
            if tree.right.pred_prob < max_left_prob:
                return False
    if isinstance(tree.left, Leaf):
        if not isinstance(tree.right, Leaf):
            max_right_prob = get_max_prob(tree.right)
            if tree.left.pred_prob < max_right_prob:
                return False
    return True


def tree_height(tree: Any) -> int:
    """
    Compute the height of a tree (AVL-style).
    Leaves have height 0, and each level adds 1.
    Returns -1 for None/empty trees.
    """
    if tree is None:
        return -1
    if isinstance(tree, Leaf):
        return 0
    return 1 + max(tree_height(tree.left), tree_height(tree.right))


def count_leaves(tree: Any) -> int:
    """
    Count the number of leaves in a tree.
    """
    if tree is None:
        return 0
    if isinstance(tree, Leaf):
        return 1
    return count_leaves(tree.left) + count_leaves(tree.right)


def colless_index(tree: Any) -> int:
    """
    Compute the Colless index for a binary tree.
    The Colless index measures tree imbalance by summing the absolute differences
    between the number of leaves in left and right subtrees for all internal nodes.
    
    Returns:
        - 0 for leaves (no internal nodes)
        - int: sum of |left_leaves - right_leaves| over all internal nodes
    """
    if isinstance(tree, Leaf):
        return 0

    if not isinstance(tree, Node):
        return 0

    # Count leaves in left and right subtrees
    left_leaves = count_leaves(tree.left)
    right_leaves = count_leaves(tree.right)

    # Current node's contribution to Colless index
    current_diff = abs(left_leaves - right_leaves)

    # Recursively compute Colless index for subtrees
    left_colless = colless_index(tree.left)
    right_colless = colless_index(tree.right)

    return current_diff + left_colless + right_colless


def normalized_colless_index(tree: Any) -> float:
    """
    Compute the Normalized Colless index for a binary tree.
    Normalizes the Colless index by dividing by the maximum possible Colless index
    for a tree with the same number of leaves.
    
    The maximum Colless index for a tree with n leaves is (n-1)(n-2)/2
    (achieved by a caterpillar/comb tree).
    
    Returns:
        - 0.0 for leaves (perfectly balanced, single leaf)
        - float: normalized value in [0, 1] where:
            * 0 = perfectly balanced tree
            * 1 = maximally imbalanced tree (caterpillar)
    """
    n = count_leaves(tree)

    # Handle edge cases
    if n <= 1:
        return 1.0

    # Compute raw Colless index
    raw_colless = colless_index(tree)

    # Maximum Colless index for n leaves (caterpillar tree)
    max_colless = (n - 1) * (n - 2) // 2

    if max_colless == 0:
        return 1.0

    # Normalize
    return raw_colless / max_colless


# Global memo for subproblem optimal objectives
# Key: Tuple of (frozenset of (feature, direction) tuples representing the path, depth)
# Value: (best_loss, pmax_best, depth_of_pmax_best, is_root_internal)
# Using OrderedDict to track insertion order for LRU cache eviction
_subproblem_optimal_objectives: OrderedDict[Tuple[frozenset, int], Tuple[float, float, int, bool]] = OrderedDict()

# Maximum cache size for subproblem optimal objectives
# When exceeded, oldest entries are removed.
# Define a sensible module-level default so imports (e.g., from run_rashomon_set.py)
# always see a defined value.
MAX_CACHE_SIZE: int = 100000
_cache_hits = 0
_cache_misses = 0


def _set_subproblem_optimal_objective(key: Tuple[frozenset, int],
                                      value: Tuple[float, float, int, bool]):
    """
    Set the optimal objective for a subproblem with cache size management.
    When cache size exceeds MAX_CACHE_SIZE, removes the oldest entry.
    
    Args:
        key: Tuple of (subproblem_path, depth)
        value: Tuple of (best_loss, pmax_best, depth_of_pmax_best, is_root_internal)
    """
    # If key already exists, move it to end (most recently used)
    if key in _subproblem_optimal_objectives:
        _subproblem_optimal_objectives.move_to_end(key)

    # Set the value
    _subproblem_optimal_objectives[key] = value

    # Remove oldest entry if cache size exceeded
    if len(_subproblem_optimal_objectives) > MAX_CACHE_SIZE:
        _subproblem_optimal_objectives.popitem(last=False)  # Remove oldest (first) item


def _get_subproblem_optimal_objective_value(
    key: Tuple[frozenset, int],
) -> Optional[Tuple[float, float, int, bool]]:
    """
    Get the optimal objective value for a subproblem key.
    Updates access order for LRU cache behavior.
    
    Args:
        key: Tuple of (subproblem_path, depth)
    
    Returns:
        Tuple of (best_loss, pmax_best, depth_of_pmax_best, is_root_internal) if found, None otherwise
    """
    global _cache_hits, _cache_misses
    if key in _subproblem_optimal_objectives:
        # Move to end to mark as recently used
        _subproblem_optimal_objectives.move_to_end(key)
        _cache_hits += 1
        return _subproblem_optimal_objectives[key]
    _cache_misses += 1
    return None


def get_subproblem_optimal_objective(subproblem_path: frozenset,
                                     depth: int) -> Optional[float]:
    """
    Get the optimal objective for a subproblem.
    Updates access order for LRU cache behavior.
    
    Args:
        subproblem_path: Frozenset of (feature, direction) tuples, e.g., frozenset({(12, '0'), (3, '1')})
                        Equivalent paths automatically map to the same key (frozensets are normalized)
    
    Returns:
        Optimal objective if found, None otherwise
    """
    result = _get_subproblem_optimal_objective_value((subproblem_path, depth))
    return result if result is None else result[0]


def get_all_subproblem_objectives() -> Dict[frozenset, float]:
    """
    Get all stored subproblem optimal objectives.
    
    Returns:
        Dictionary mapping subproblem paths (frozensets) to optimal objectives
    """
    return _subproblem_optimal_objectives.copy()


def path_to_frozenset(path: str) -> frozenset:
    """
    Convert a path string to a frozenset representation.
    For backward compatibility and string-to-frozenset conversion.
    
    Args:
        path: Path string like "1:0|2:0" or "|1:0|2:0"
    
    Returns:
        Frozenset of (feature, direction) tuples
    """
    if not path:
        return frozenset()

    # Remove leading | if present
    normalized = path.lstrip('|')

    if not normalized:
        return frozenset()

    # Split by | and parse feature:direction pairs
    parts = normalized.split('|')
    pairs = []

    for part in parts:
        if ':' in part:
            try:
                feature_str, direction = part.split(':', 1)
                feature = int(feature_str)
                pairs.append((feature, direction))
            except ValueError:
                # Invalid format, skip
                continue

    return frozenset(pairs)


def frozenset_to_path(path_set: frozenset) -> str:
    """
    Convert a frozenset path to a string representation.
    For backward compatibility and debugging.
    
    Args:
        path_set: Frozenset of (feature, direction) tuples
    
    Returns:
        Path string like "1:0|2:0" (sorted for consistency)
    """
    if not path_set:
        return ''

    # Sort pairs for consistent string representation
    pairs = sorted(path_set)
    return '|'.join(f'{f}:{d}' for f, d in pairs)


# --- Main algorithms ---
def OptFallingTree(X: np.ndarray,
                   y: np.ndarray,
                   row_idx: np.ndarray,
                   d: int,
                   lam: float,
                   features: Iterable[int],
                   n=None,
                   path=None,
                   rule_list_mode=False,
                   enable_falling_constraint: bool = True,
                   **kwargs) -> Tuple[float, Any, float, int]:
    """
    Returns (best_loss, best_tree, p_max) where p_max is maximum leaf positive-proportion.
    
    Args:
        X: binarized feature matrix (n_samples, n_features)
        y: label vector (n_samples,)
        row_idx: indices of rows in current subproblem
    """

    if "branching_cost" in kwargs:
        lam = 0
        branching_cost = kwargs["branching_cost"]
        if "min_support" in kwargs:
            min_support = kwargs["min_support"]
        else:
            min_support = 0.05  # each leaf must contain at least 5% of the data
    else:
        min_support = 0
        branching_cost = 0

    if path is None:
        path = frozenset()
    # baseline: make a leaf
    L_leaf, p_leaf = leaf_cost(X, y, row_idx, lam, n)
    best_loss = L_leaf
    best_tree = Leaf(pred_prob=p_leaf, objective=L_leaf)
    pmax_best = p_leaf
    depth_of_pmax_best = d
    is_root_internal = False
    is_root_internal_best = is_root_internal

    # Store optimal objective in global subproblem memo
    # Path is already a frozenset, so it's automatically normalized
    cached_value = _get_subproblem_optimal_objective_value((path, d))
    if cached_value is not None and best_loss < cached_value[0]:
        _set_subproblem_optimal_objective((path, d), (best_loss, pmax_best, depth_of_pmax_best, is_root_internal_best))

    if d == 0:
        result = (best_loss, best_tree, pmax_best, depth_of_pmax_best)
        _set_subproblem_optimal_objective((path, d), (best_loss, pmax_best, depth_of_pmax_best, is_root_internal_best))
        return result

    # Extract used features from path (O(k) where k is path length)
    used_features = {feature for feature, _ in path}

    # Pre-compute leaf losses (upper bounds) for sorting features by promise
    # Note: leaf_loss is an upper bound, so we can't prune based on it alone,
    # but we can use it to order features and try promising ones first
    feature_candidates = []
    for j in features:
        if j in used_features:
            continue

        # Split using numpy masks
        mask = X[row_idx, j] == 0
        left_idx = row_idx[mask]
        right_idx = row_idx[~mask]

        if len(left_idx) / n <= min_support or len(right_idx) / n <= min_support:
            continue

        left_prob = y[left_idx].mean()
        right_prob = y[right_idx].mean()
        if left_prob < 0.5 and right_prob < 0.5:
            continue
        if left_prob >= 0.5 and right_prob >= 0.5:
            continue

        # Compute leaf loss (upper bound) for sorting
        L_L_upper, _ = leaf_cost(X, y, left_idx, lam, n)
        L_R_upper, _ = leaf_cost(X, y, right_idx, lam, n)
        leaf_loss_upper = L_L_upper + L_R_upper

        feature_candidates.append((leaf_loss_upper, j, left_idx, right_idx))

    # Sort features by upper bound (best first) to try promising features first
    # This improves pruning effectiveness since best_loss improves faster
    feature_candidates.sort(key=lambda x: x[0])

    # iterate features (now sorted by upper bound)
    for leaf_loss_upper, j, left_idx, right_idx in feature_candidates:
        # if leaf_loss_upper >= best_loss:
        #     break
        # Check memo for optimal objectives before recursing
        new_path_left = path | frozenset({(j, '0')})
        new_path_right = path | frozenset({(j, '1')})

        # If we already know the optimal objectives from memo, use them for early pruning
        if (new_path_left, d - 1) in _subproblem_optimal_objectives and (new_path_right,
                                                                         d - 1) in _subproblem_optimal_objectives:
            continue
            # L_L, pmax_L, depth_of_pmax_L = _subproblem_optimal_objectives[(new_path_left, d - 1)]
            # L_R, pmax_R, depth_of_pmax_R = _subproblem_optimal_objectives[(new_path_right, d - 1)]
        else:
            if rule_list_mode:
                left_prob = y[left_idx].mean()
                right_prob = y[right_idx].mean()
                if left_prob >= 0.5 and right_prob < 0.5:  # only recurse on the right subtree
                    L_R, t_R, pmax_R, depth_of_pmax_R = OptFallingTree(X, y, right_idx, d - 1, lam, features, n,
                                                                       new_path_right, rule_list_mode,
                                                                       enable_falling_constraint, **kwargs)
                    L_L, t_L, pmax_L, _ = OptFallingTree(X, y, left_idx, 0, lam, features, n, new_path_left,
                                                         rule_list_mode, enable_falling_constraint, **kwargs)  # this is a leaf
                    depth_of_pmax_L = d - 1
                else:
                    L_L, t_L, pmax_L, depth_of_pmax_L = OptFallingTree(X, y, left_idx, d - 1, lam, features, n,
                                                                       new_path_left, rule_list_mode,
                                                                       enable_falling_constraint, **kwargs)
                    L_R, t_R, pmax_R, _ = OptFallingTree(X, y, right_idx, 0, lam, features, n, new_path_right,
                                                         rule_list_mode, enable_falling_constraint, **kwargs)
                    depth_of_pmax_R = d - 1
            else:
                L_R, t_R, pmax_R, depth_of_pmax_R = OptFallingTree(X, y, right_idx, d - 1, lam, features, n,
                                                                   new_path_right, rule_list_mode,
                                                                   enable_falling_constraint, **kwargs)
                L_L, t_L, pmax_L, depth_of_pmax_L = OptFallingTree(X, y, left_idx, d - 1, lam, features, n,
                                                                   new_path_left, rule_list_mode,
                                                                   enable_falling_constraint, **kwargs)
                
            _set_subproblem_optimal_objective(
                (new_path_left, d - 1),
                (L_L, pmax_L, depth_of_pmax_L, isinstance(t_L, Node)))
            _set_subproblem_optimal_objective(
                (new_path_right, d - 1),
                (L_R, pmax_R, depth_of_pmax_R, isinstance(t_R, Node)))
        L_split = L_L + L_R

        if isinstance(t_L, Node) and isinstance(t_R, Node):
            L_split += branching_cost
            is_root_internal = True
        else:
            is_root_internal = False

        if L_split >= best_loss:
            continue

        # Falling tree definition 1
        # if isinstance(t_L, Leaf):
        #     p_plus_L = t_L.pred_prob
        #     if not isinstance(t_R, Leaf) and p_plus_L < pmax_R:
        #         continue

        # if isinstance(t_R, Leaf):
        #     p_plus_R = t_R.pred_prob
        #     if not isinstance(t_L, Leaf) and p_plus_R < pmax_L:
        #         continue

        # Falling tree definition 2
        if enable_falling_constraint:
            if pmax_L > pmax_R:
                if depth_of_pmax_L < depth_of_pmax_R:
                    continue
            if pmax_R > pmax_L:
                if depth_of_pmax_R < depth_of_pmax_L:
                    continue

        if pmax_L > pmax_R:
            depth_of_pmax_cand = depth_of_pmax_L
            pmax_cand = pmax_L
        else:
            depth_of_pmax_cand = depth_of_pmax_R
            pmax_cand = pmax_R

        cand = Node(feature=j, left=t_L, right=t_R, objective=L_split)

        if L_split < best_loss:
            best_loss = L_split
            best_tree = cand
            pmax_best = pmax_cand
            depth_of_pmax_best = depth_of_pmax_cand
            is_root_internal_best = is_root_internal

    # Update global subproblem memo with optimal objective
    cached_value = _get_subproblem_optimal_objective_value((path, d))
    if cached_value is None or best_loss < cached_value[0]:
        _set_subproblem_optimal_objective((path, d), (best_loss, pmax_best, depth_of_pmax_best, is_root_internal_best))

    # Cache all subtree solutions from the optimal tree
    # This allows OptFallingRset to reuse optimal subproblem solutions without recomputing
    # if not isinstance(best_tree, Leaf):
    #     cache_subtree_solutions(best_tree, path, d, n)

    result = (best_loss, best_tree, pmax_best, depth_of_pmax_best)
    return result


def OptFallingRset(X: np.ndarray,
                   y: np.ndarray,
                   row_idx: np.ndarray,
                   d: int,
                   lam: float,
                   B: float,
                   features: Iterable[int],
                   n=None,
                   path=None,
                   enable_falling_constraint: bool = True,
                   rule_list_mode: bool = False,
                   use_heap: bool = True,
                   **kwargs) -> List[Tuple[Any, float, int, float]]:
    """
    Returns a sorted list R of tuples (tree, p_max, obj) of trees with objective <= B, sorted by objective.
    If use_heap=True, uses a heap internally to maintain sorted order without repeated sorting operations.
    If use_heap=False, collects all trees in a list and sorts at the end.
    Args:
        X: binarized feature matrix (n_samples, n_features)
        y: label vector (n_samples,)
        row_idx: indices of rows in current subproblem
    """

    if "branching_cost" in kwargs:
        lam = 0
        branching_cost = kwargs["branching_cost"]
        if "min_support" in kwargs:
            min_support = kwargs["min_support"]
        else:
            min_support = 0.05  # each leaf must contain at least 5% of the data
    else:
        min_support = 0
        branching_cost = 0

    if path is None:
        path = frozenset()

    # Initialize storage based on use_heap flag
    if use_heap:
        # Use a heap to maintain sorted order as we add items
        # Heap stores (obj, p_max, depth_of_pmax, tie_breaker, tree) for sorting by obj, with tie_breaker to avoid Node comparison
        R_heap = []
        tie_breaker = 0  # Counter to break ties when obj and p_max are equal
    else:
        # Use a list to collect all trees, will sort at the end
        R_list = []

    # base: leaf (always consider making a leaf at any depth)
    L_leaf, p_leaf = leaf_cost(X, y, row_idx, lam, n)
    depth_of_pmax_leaf = d

    if L_leaf <= B:
        if use_heap:
            heapq.heappush(R_heap,
                           (L_leaf, p_leaf, depth_of_pmax_leaf, tie_breaker,
                            Leaf(pred_prob=p_leaf, objective=L_leaf)))
            tie_breaker += 1
        else:
            R_list.append((Leaf(pred_prob=p_leaf, objective=L_leaf), p_leaf, depth_of_pmax_leaf, L_leaf))

    if d == 0:
        # At depth 0, we can only make leaves, so return what we have
        if len(row_idx) / n <= min_support:
            return []
        if use_heap:
            if len(R_heap) == 0:
                return []
            # Convert heap to sorted list: (obj, p_max, depth_of_pmax, tie_breaker, tree[, profile])
            return [(tree, p_max, depth_of_pmax, obj)
                    for obj, p_max, depth_of_pmax, tie_breaker, tree in heapq.nsmallest(len(R_heap), R_heap)]
        else:
            if len(R_list) == 0:
                return []
            # Sort the list by objective
            R_list.sort(key=lambda x: x[3])  # Sort by obj (4th element)
            return R_list
    # Extract used features from path for filtering
    used_features = {feature for feature, _ in path}

    # Pre-compute leaf_losses (upper bounds) for sorting features by promise
    # Note: leaf_loss is an upper bound, so we can't prune based on it alone,
    # but we can use it to order features and try promising ones first
    feature_candidates = []
    for j in features:
        if j in used_features:
            continue

        # Split using numpy masks
        mask = X[row_idx, j] == 0
        left_idx = row_idx[mask]
        right_idx = row_idx[~mask]

        # Skip if partition supports are too small
        if len(left_idx) / n <= min_support or len(right_idx) / n <= min_support:
            continue

        # Compute leaf loss (upper bound) for sorting
        leaf_loss = leaf_cost(X, y, left_idx, lam, n)[0] + leaf_cost(X, y, right_idx, lam, n)[0]

        # Check memo early to potentially prune before expensive computation
        new_path_left = path | frozenset({(j, '0')})
        new_path_right = path | frozenset({(j, '1')})

        feature_candidates.append((leaf_loss, j, left_idx, right_idx, new_path_left, new_path_right))

    # Sort features by upper bound (best first) to improve pruning effectiveness
    feature_candidates.sort(key=lambda x: x[0])

    # iterate features (now sorted and pre-filtered)
    for leaf_loss, j, left_idx, right_idx, new_path_left, new_path_right in feature_candidates:
        # best tree losses for left and right via OptFallingTree
        # Create a copy of features without j for recursive calls
        features_without_j = [f for f in features if f != j]

        # Check global subproblem memo first (paths are already frozensets, so automatically normalized)
        left_result = _get_subproblem_optimal_objective_value((new_path_left, d - 1))
        right_result = _get_subproblem_optimal_objective_value((new_path_right, d - 1))
        if left_result is not None and right_result is not None:
            # Both subproblems already solved and cached
            L_L_star, pmax_L_star, depth_of_pmax_L_star, is_left_internal = left_result
            L_R_star, pmax_R_star, depth_of_pmax_R_star, is_right_internal = right_result
        else:
            # Compute if not in memo
            # Note: OptFallingTree only explores paths that can improve the optimal solution,
            # so many subproblems needed by OptFallingRset (which explores ALL paths within budget)
            # may not be cached. This is expected behavior.
            L_L_star, t_L_star, pmax_L_star, depth_of_pmax_L_star = OptFallingTree(
                X,
                y,
                left_idx,
                d - 1,
                lam,
                features_without_j,
                n=n,
                path=new_path_left,
                rule_list_mode=rule_list_mode,
                enable_falling_constraint=enable_falling_constraint,
                **kwargs,
            )
            L_R_star, t_R_star, pmax_R_star, depth_of_pmax_R_star = OptFallingTree(
                X,
                y,
                right_idx,
                d - 1,
                lam,
                features_without_j,
                n=n,
                path=new_path_right,
                rule_list_mode=rule_list_mode,
                enable_falling_constraint=enable_falling_constraint,
                **kwargs,
            )

            is_left_internal = isinstance(t_L_star, Node)
            is_right_internal = isinstance(t_R_star, Node)

            _set_subproblem_optimal_objective(
                (new_path_left, d - 1),
                (L_L_star, pmax_L_star, depth_of_pmax_L_star, is_left_internal),
            )
            _set_subproblem_optimal_objective(
                (new_path_right, d - 1),
                (L_R_star, pmax_R_star, depth_of_pmax_R_star, is_right_internal),
            )

        # For pruning we use a lower bound on the parent objective:
        # subproblem objectives already include any branching costs within each subtree.
        # We defer adding the potential branching_cost at this new root until we combine
        # explicit left/right trees later, when we know if both children are internal.
        L_split = L_L_star + L_R_star
        if is_left_internal and is_right_internal:
            L_split += branching_cost

        if L_split <= B:
            # proceed to enumerate R-sets recursively (only when split could be within budget)
            # Recursive calls already return sorted lists, so we can use them directly
            R_L = OptFallingRset(X,
                                 y,
                                 left_idx,
                                 d - 1,
                                 lam,
                                 B - L_R_star,
                                 features_without_j,
                                 n=n,
                                 path=new_path_left,
                                 rule_list_mode=rule_list_mode,
                                 enable_falling_constraint=enable_falling_constraint,
                                 use_heap=use_heap,
                                 **kwargs)
            R_R = OptFallingRset(X,
                                 y,
                                 right_idx,
                                 d - 1,
                                 lam,
                                 B - L_L_star,
                                 features_without_j,
                                 n=n,
                                 path=new_path_right,
                                 rule_list_mode=rule_list_mode,
                                 enable_falling_constraint=enable_falling_constraint,
                                 use_heap=use_heap,
                                 **kwargs)
            # R_L and R_R are already sorted by objective from recursive calls
            # iterate combinations with early termination optimization
            already_seen_pairs = set()
            # if d == 4:
            #     R_lt = tqdm(R_L)
            # else:
            #     R_lt = R_L
            min_objR = R_R[0][3] if R_R else None
            for t_L, pL_max, depth_of_pmax_L, objL in R_L:
                # Early termination: if objL + min_objR > B, no more valid pairs exist
                if min_objR is not None and objL + min_objR > B:
                    break
                # if isinstance(t_L, Leaf):

                for t_R, pR_max, depth_of_pmax_R, objR in R_R:
                    if (t_L, t_R) in already_seen_pairs:
                        continue
                    already_seen_pairs.add((t_L, t_R))
                    already_seen_pairs.add((t_R, t_L))
                    total_obj = objL + objR
                    if isinstance(t_L, Node) and isinstance(t_R, Node):
                        total_obj += branching_cost
                    if total_obj > B:
                        break  # Early termination: since R_R is sorted, all remaining objR will also exceed B
                    if isinstance(t_L, Leaf) and isinstance(t_R, Leaf):
                        if t_L.pred_prob < 0.5 and t_R.pred_prob < 0.5:
                            continue
                        if t_L.pred_prob >= 0.5 and t_R.pred_prob >= 0.5:
                            continue
                    # apply falling constraints similar to pseudocode
                    # Check constraints first before creating Node to save memory
                    if enable_falling_constraint:
                        # Falling tree definition 2
                        # Determine which subtree has the max probability and check depth constraint
                        # Note: depth_of_pmax values are relative to the subtree root (depth d-1)
                        # The constraint compares these depths directly (both relative to same level)
                        if pL_max > pR_max:
                            # Left has higher max probability
                            # Constraint: depth_of_pmax_L must be >= depth_of_pmax_R
                            # (higher prob should be at shallower depth, i.e., larger depth value)
                            if depth_of_pmax_L < depth_of_pmax_R:
                                continue  # Skip: violates falling constraint
                            depth_of_pmax_cand = depth_of_pmax_L
                            pmax_cand = pL_max
                        elif pR_max > pL_max:
                            # Right has higher max probability
                            # Constraint: depth_of_pmax_R must be >= depth_of_pmax_L
                            if depth_of_pmax_R < depth_of_pmax_L:
                                continue  # Skip: violates falling constraint
                            depth_of_pmax_cand = depth_of_pmax_R
                            pmax_cand = pR_max
                        else:
                            # pL_max == pR_max: both subtrees have same max probability
                            # No constraint violation possible, use either (prefer deeper)
                            if depth_of_pmax_L >= depth_of_pmax_R:
                                depth_of_pmax_cand = depth_of_pmax_L
                            else:
                                depth_of_pmax_cand = depth_of_pmax_R
                            pmax_cand = pL_max  # or pR_max, they're equal
                    else:
                        # No falling constraints: use the max probability from either subtree
                        if pL_max >= pR_max:
                            depth_of_pmax_cand = depth_of_pmax_L
                            pmax_cand = pL_max
                        else:
                            depth_of_pmax_cand = depth_of_pmax_R
                            pmax_cand = pR_max

                    # Create node only after all constraints have been verified, in order to save memory
                    t_candidate = Node(feature=j, left=t_L, right=t_R, objective=total_obj)
                    if use_heap:
                        heapq.heappush(R_heap, (total_obj, pmax_cand, depth_of_pmax_cand,
                                                tie_breaker, t_candidate))
                        tie_breaker += 1
                    else:
                        R_list.append((t_candidate, pmax_cand, depth_of_pmax_cand, total_obj))

                    # Falling tree definition 1
                    # if isinstance(t_R, Leaf):
                    #     pR_plus = t_R.pred_prob
                    #     if pR_plus > pL_max and not isinstance(t_L, Leaf):
                    #         # Add to heap: (obj, p_max, tie_breaker, tree) for sorting by objective
                    #         heapq.heappush(R_heap, (total_obj, pR_plus, tie_breaker, t_candidate))
                    #         tie_breaker += 1
                    #         continue
                    # if isinstance(t_L, Leaf):
                    #     pL_plus = t_L.pred_prob
                    #     if pL_plus > pR_max and not isinstance(t_R, Leaf):
                    #         heapq.heappush(R_heap, (total_obj, pL_plus, tie_breaker, t_candidate))
                    #         tie_breaker += 1
                    #         continue
        else:
            continue

    # Convert to sorted list
    if use_heap:
        # Convert heap to sorted list: (obj, p_max, depth_of_pmax, tie_breaker, tree)
        # heapq.nsmallest extracts all elements in sorted order
        # Note: depth_of_pmax is stored in the heap, so we extract it correctly
        result = [(tree, p_max, depth_of_pmax, obj)
                  for obj, p_max, depth_of_pmax, tie_breaker, tree in heapq.nsmallest(len(R_heap), R_heap)]
    else:
        # Sort the list by objective (post-hoc sorting)
        R_list.sort(key=lambda x: x[3])  # Sort by obj (4th element)
        result = R_list
    return result


# --- Pretty-printing tree ---
def print_tree(t: Any, depth: int = 0):
    ind = "  " * depth
    if isinstance(t, Leaf):
        print(f"{ind}Leaf(pred_prob={t.pred_prob:.3f}, objective={t.objective:.3f})")
    else:
        print(f"{ind}Node(feature={t.feature})")
        print_tree(t.left, depth + 1)
        print_tree(t.right, depth + 1)


# --- Small example test ---
if __name__ == "__main__":

    # Load Compas dataset
    compas_path = "/usr/xtmp/vb97/FRL_Rashomon_Set/falling-models/data/benchmark/fico.csv"

    print("Loading Compas dataset...")
    df = pd.read_csv(compas_path)

    # Separate features and labels
    # Last column is two_year_recid (label)
    # Features: sex=female, age, juv_fel_count, juv_misd_count, juvenile_crimes, priors_count, current_charge_degree=felony
    feature_cols = df.columns[:-1]
    label_col = df.columns[-1]

    X = df[feature_cols]
    y = df[label_col]

    print(f"Dataset shape: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Feature columns: {list(feature_cols)}")
    print(f"Label column: {label_col}")
    print(f"Positive label proportion: {y.mean():.3f}")

    # Binarize numeric features (convert to 0/1 based on median threshold)
    # Features that are already binary: sex=female (0), current_charge_degree=felony (6)
    # Features to binarize: age (1), juv_fel_count (2), juv_misd_count (3), juvenile_crimes (4), priors_count (5)
    # X_binary = X.copy().astype(float)
    # threshold guess
    df, thresholds, header, threshold_guess_time = binarize_dataset(df, num_estimators=100)
    # df.to_csv('compas_binarized.csv')
    # df.to_csv("datasets/compas_binarized.csv", index=False)
    # for i in range(X.shape[1]):
    #     if i not in [0, 6]:  # Skip already binary features
    #         median_val = np.median(X[:, i])
    #         X_binary[:, i] = (X[:, i] > median_val).astype(float)

    # Convert to numpy arrays
    X = df.iloc[:, :-1].values.astype(int)
    y = df.iloc[:, -1].values.astype(int)
    print(f"Dataset shape: {X.shape[0]} samples, {X.shape[1]} features")

    features = list(range(X.shape[1]))
    lam = 0.005
    eps = 0.02
    depth = 4  # Reasonable depth for this dataset
    enable_falling_constraint = True
    use_heap = True  # Use heap for storing trees in Rashomon set
    rule_list_mode_for_optimal = False
    MAX_CACHE_SIZE = 100000  # Default value, can be adjusted
    _subproblem_optimal_objectives.clear()
    kwargs = {"branching_cost": 0.02, "min_support": 0.02}
    # kwargs = {}
    print(f"\nRunning OptFallingTree on Compas dataset (depth={depth})...")
    n = X.shape[0]
    row_idx = np.arange(n)

    start_time = time.time()
    best_loss, best_tree, pmax, depth_of_pmax = OptFallingTree(X,
                                                               y,
                                                               row_idx,
                                                               depth,
                                                               lam,
                                                               features,
                                                               n=n,
                                                               enable_falling_constraint=enable_falling_constraint,
                                                               rule_list_mode=rule_list_mode_for_optimal,
                                                               **kwargs)
    elapsed_time = time.time() - start_time
    is_best_tree_falling = is_falling_tree(best_tree)
    print(f"Is best tree falling?: {is_best_tree_falling}")

    print(f"Best loss: {best_loss:.4f}, pmax: {pmax:.4f}, depth_of_pmax: {depth_of_pmax}")
    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    print("\nBest tree:")
    print_tree(best_tree)
    print(f"Normalized colless index of best tree: {normalized_colless_index(best_tree):.4f} ")
    print(f"Subproblem memo size: {len(_subproblem_optimal_objectives)}")
    print(f"\nComputing R-set with budget = best_loss (should include at least the best tree):")
    start_time = time.time()
    R = OptFallingRset(X,
                       y,
                       row_idx,
                       depth,
                       lam,
                       B=best_loss * (1 + eps),
                       features=features,
                       n=n,
                       enable_falling_constraint=enable_falling_constraint,
                       use_heap=use_heap,
                       rule_list_mode=rule_list_mode_for_optimal,
                       **kwargs)
    elapsed_time = time.time() - start_time
    print(f"Subproblem memo size after OptFallingRset: {len(_subproblem_optimal_objectives)}")
    print(f"Found {len(R)} trees in R-set with budget <= {best_loss+eps:.4f}")
    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    normalized_colless_indices = [normalized_colless_index(t) for (t, p, depth_of_pmax_leaf, obj) in R]
    # print(f"Normalized colless indices: {normalized_colless_indices}")
    print(f"Average normalized colless index: {np.mean(normalized_colless_indices):.4f}")
    print(f"Min normalized colless index: {np.min(normalized_colless_indices):.4f}")
    print(f"Max normalized colless index: {np.max(normalized_colless_indices):.4f}")
    print(f"Cache hits: {_cache_hits}, cache misses: {_cache_misses}")
    # for t, p, depth_of_pmax_leaf, obj in R:  # R is already sorted by objective
    #     if normalized_colless_index(t) >= 0.0: # TODO: remove this
    #         print(f"--- tree obj={obj:.4f}, pmax={p:.4f}, depth_of_pmax_leaf={depth_of_pmax_leaf} ---")
    #         print_tree(t)
    #         is_tree_falling = is_falling_tree(t)
    #         print(f"Is tree falling?: {is_tree_falling}")
    # with open("profile.log", "w") as f:
    #     stats = pstats.Stats("out.prof", stream=f)
    #     stats.strip_dirs()
    #     stats.sort_stats(pstats.SortKey.CUMULATIVE)
    #     # stats.print_stats("OptFallingRset")
    #     stats.print_callers("OptFallingRset")
    #     stats.print_callees("OptFallingRset")

    # save the Rashomon set to a file
    with open(
            f"Rashomon_set_compas_lambda_{lam}_depth_{depth}_eps_{eps}_falling_constraint_{enable_falling_constraint}.pkl",
            "wb") as f:
        pkl.dump(R, f)
