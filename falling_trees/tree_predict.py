import numpy as np
import pandas as pd


def predict_proba_falling_tree(tree, X):
    """Get predicted probabilities from a falling tree, preserving input order."""
    if isinstance(X, pd.DataFrame):
        X = X.values

    n_samples = X.shape[0]
    probs = np.empty(n_samples)

    def recurse(node, idx):
        if node.is_leaf():
            probs[idx] = node.pred_prob
            return

        feat = node.feature
        left_idx = idx[X[idx, feat] == 0]
        right_idx = idx[X[idx, feat] == 1]

        if len(left_idx) > 0:
            recurse(node.left, left_idx)
        if len(right_idx) > 0:
            recurse(node.right, right_idx)

    recurse(tree, np.arange(n_samples))
    return probs


def predict_falling_tree(tree, X, threshold=0.5):
    """Predict binary labels from a falling tree using a threshold."""
    probs = predict_proba_falling_tree(tree, X)
    return (probs > threshold).astype(int)
