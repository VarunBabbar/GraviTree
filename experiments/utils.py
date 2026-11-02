import sys
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Iterable, Set

# Add parent directory to path so we can import falling_trees and frame
# script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = "/usr/xtmp/vb97/FRL_Rashomon_Set/falling-models"
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import argparse
from falling_trees.binarize_dataset import binarize_dataset
from falling_trees import frl_rashomon_set_alg
from falling_trees.frl_rashomon_set_alg import Node, Leaf, OptFallingTree, OptFallingRset, tree_obj, _subproblem_optimal_objectives, print_tree
from frame.rashomon_sets import FRLRashomonSet


def expected_decision_sparsity_falling_tree(node, dataset: np.ndarray):
    """
    Compute the expected decision sparsity of a node given a dataset.
    """
    # Use is_leaf() so trees from frl_rashomon_set_alg_falling_constraint_v2 match
    # (distinct Leaf classes from frl_rashomon_set_alg break isinstance checks).
    if node.is_leaf():
        return 0
    
    current_feature = node.feature
    left_dataset = dataset[dataset[:, current_feature] == 0]
    right_dataset = dataset[dataset[:, current_feature] == 1]
    eps = 1e-6
    return (len(left_dataset) * (1+expected_decision_sparsity_falling_tree(node.left, left_dataset)) + len(right_dataset) * (1+expected_decision_sparsity_falling_tree(node.right, right_dataset))) / (len(dataset)+eps)


def expected_decision_sparsity_frame_frl(frl, dataset: np.ndarray):
    """
    Compute the expected decision sparsity of a FRAME FRL given a dataset.
    
    Expected decision sparsity = expected number of features checked to evaluate
    whether a point is captured by the rule list.
    
    For each point, we evaluate rules in order:
    - For each rule, we check ALL features in that rule's antecedent
    - If the point matches a rule, we stop (count all features checked so far)
    - If the point doesn't match, we continue to the next rule
    - If no rule matches, the point goes to the else clause (count all features in all rules)
    
    FRAME may add complement features during preprocessing. The rule_list contains
    feature indices that refer to the expanded feature space.
    
    Parameters:
    -----------
    frl : FallingRuleList
        A FRAME falling rule list with rule_list attribute
    dataset : np.ndarray or pd.DataFrame
        The dataset (may need to be expanded with complement features)
        Should be a boolean array
    
    Returns:
    --------
    float
        Expected decision sparsity (average number of features checked per sample)
    """
    # Convert to numpy array if needed
    is_dataframe = isinstance(dataset, pd.DataFrame)
    if is_dataframe:
        dataset = dataset.values
    
    # Ensure boolean type
    dataset = dataset.astype(bool)
    
    # Check if FRAME used complement features
    if hasattr(frl, 'included_complement') and frl.included_complement:
        # Find the maximum feature index used in any rule
        max_feature_idx = -1
        for antecedent, _ in frl.rule_list:
            if len(antecedent) > 0:
                max_feature_idx = max(max_feature_idx, max(antecedent))
        
        n_current_features = dataset.shape[1]
        n_samples = dataset.shape[0]
        
        # If the dataset already has enough features (2x original), don't expand
        # Otherwise, expand to match FRAME's feature space
        if max_feature_idx >= n_current_features:
            # Need to expand: dataset doesn't have complement features yet
            n_original_features = n_current_features
            
            # Create expanded dataset: original features + complements
            # Features 0 to n-1 are original, features n to 2n-1 are complements
            expanded_dataset = np.zeros((n_samples, 2 * n_original_features), dtype=bool)
            expanded_dataset[:, :n_original_features] = dataset
            expanded_dataset[:, n_original_features:] = ~dataset
            
            dataset_to_use = expanded_dataset
        else:
            # Dataset already has the right number of features (or more)
            dataset_to_use = dataset
    else:
        dataset_to_use = dataset
    
    # Compute expected decision sparsity
    # For each sample, count features checked until we find a matching rule
    n_samples = dataset_to_use.shape[0]
    total_features_checked = 0.0
    
    # Track which samples have been captured by previous rules
    already_captured = np.zeros(n_samples, dtype=bool)
    
    # Track cumulative number of features checked so far (across all previous rules)
    features_checked_so_far = 0
    
    for antecedent, _ in frl.rule_list:
        if len(antecedent) == 0:
            # Default/else rule - captures all remaining samples
            # For these samples, we've checked all features in all previous rules
            remaining_samples = ~already_captured
            n_remaining = np.sum(remaining_samples)
            # Add features checked for all remaining samples (all features in all rules)
            total_features_checked += n_remaining * features_checked_so_far
            already_captured = np.ones(n_samples, dtype=bool)  # All captured now
            break
        
        # Check which samples satisfy this antecedent
        # All features in the antecedent must be 1
        satisfies_antecedent = np.all(dataset_to_use[:, antecedent] == 1, axis=1)
        newly_captured = satisfies_antecedent & ~already_captured
        n_newly_captured = np.sum(newly_captured)
        
        # For samples that match this rule:
        # - We checked all features in previous rules (features_checked_so_far)
        # - We checked all features in current rule (len(antecedent))
        # Total = features_checked_so_far + len(antecedent)
        total_features_checked += n_newly_captured * (features_checked_so_far + len(antecedent))
        
        # For samples that don't match this rule:
        # - We still checked all features in this rule (len(antecedent))
        # - They'll be checked in the next rule
        # So we add len(antecedent) to features_checked_so_far for the next iteration
        
        # Update cumulative features checked (for next rule)
        features_checked_so_far += len(antecedent)
        
        # Mark these samples as captured
        already_captured |= newly_captured
    
    # Handle any remaining uncaptured samples (shouldn't happen if last rule is default)
    remaining_samples = ~already_captured
    n_remaining = np.sum(remaining_samples)
    if n_remaining > 0:
        # These would be captured by the default rule
        # We've checked all features in all rules
        total_features_checked += n_remaining * features_checked_so_far
    
    return total_features_checked / n_samples



def extract_frl_rules(frl):
    """
    Return leaves in execution order.
    """
    rules = []

    while isinstance(frl, Node):
        leaf, frl = next_leaf_and_rest(frl)
        rules.append(leaf)

    return rules



# def number_of_terms(node, flag = True):
#     frl = convert_falling_tree_to_falling_rule_list(node)
#     rules = extract_frl_rules(frl)
#     return sum([len(i) for i in minimal_conditions(rules)])


def is_leaf(x):
    return isinstance(x, Leaf)

def ensure_frl(x):
    if x is None:
        return None
    if isinstance(x, Leaf):
        return Node(
            feature=None,
            left=x,
            right=None,
            objective=0,
        )
    return x


def next_leaf_and_rest(frl: Node):
    if isinstance(frl.left, Leaf):
        return frl.left, frl.right
    if isinstance(frl.right, Leaf):
        return frl.right, frl.left
    raise ValueError("Invalid FRL")

def frl_sparsity(frl):
    """Old sparsity measure for FRAME FRLs"""
    total_terms = 0
    for antecedent, _ in frl.rule_list:
        total_terms += len(antecedent)
    return total_terms

def frl_accuracy(frl, X, y):
    y_pred = frl.predict(X)
    return accuracy_score(y, y_pred)

def compute_tree_test_loss_threshold(tree, X, y, threshold=0.5):
    y_pred = predict_falling_tree(tree, X, threshold)
    return np.mean(y_pred != y)


from falling_trees.tree_predict import predict_falling_tree, predict_proba_falling_tree


def predict_frl(frl, X, threshold=None):
    """
    Predict binary labels from a FRAME FRL using a threshold.
    
    Parameters:
    -----------
    frl : FallingRuleList
        A FRAME falling rule list
    X : np.ndarray or pd.DataFrame
        Input data (boolean array)
    threshold : float, optional
        Threshold for binary prediction. If predicted probability > threshold, predict 1, else 0.
        If None, uses the default threshold from frl.predict() (1 / (1 + frl.w))
    
    Returns:
    --------
    np.ndarray
        Array of binary predictions (0 or 1)
    """
    return frl.predict(X, threshold=threshold)


def evaluate_sparsity_cost(tree):
    if tree.is_leaf():
        return 1
    else:
        return evaluate_sparsity_cost(tree.left) + evaluate_sparsity_cost(tree.right)

