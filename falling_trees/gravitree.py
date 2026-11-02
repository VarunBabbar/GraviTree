from sklearn.base import BaseEstimator, ClassifierMixin
import tempfile
import os
import json
import copy
import numpy as np
import pandas as pd

import falling_trees.gravitree_cpp as gravitree_cpp
from falling_trees.binarize_dataset import binarize_dataset, cut
from falling_trees.tree_predict import predict_proba_falling_tree


class GraviTree(ClassifierMixin, BaseEstimator):

    def __init__(self, eps=0.02, depth=5, min_support=0.02, branching_cost=0.01,
                 binarize=False, num_estimators=30, max_thresholds_per_feat=4):
        self.eps = eps
        self.depth = depth
        self.min_support = min_support
        self.branching_cost = branching_cost
        self.binarize = binarize
        self.num_estimators = num_estimators
        self.max_thresholds_per_feat = max_thresholds_per_feat

    def _as_frame(self, X):
        if isinstance(X, pd.DataFrame):
            return X.copy()
        X = np.asarray(X)
        columns = [f"feature_{i}" for i in range(X.shape[1])]
        return pd.DataFrame(X, columns=columns)

    def _is_binary(self, X):
        values = np.asarray(X)
        if values.size == 0:
            return True
        return np.isin(values, [0, 1, 0.0, 1.0]).all()

    def _prepare_training_data(self, X, y):
        """Binarize continuous features when ``binarize`` is set."""
        X = self._as_frame(X)
        y = np.asarray(y)
        self.input_feature_names_ = list(X.columns)
        if self.binarize and not self._is_binary(X):
            labeled = X.copy()
            labeled["__label__"] = y
            binned, thresholds, header, _ = binarize_dataset(
                labeled,
                num_estimators=self.num_estimators,
                max_thresholds_per_feat=self.max_thresholds_per_feat,
            )
            self.binarized_ = True
            self.thresholds_ = thresholds
            self.header_ = list(header)
            X = binned.drop(columns=["__label__"]).astype(int)
            y = binned["__label__"].astype(int).to_numpy()
        else:
            self.binarized_ = False
            self.thresholds_ = None
            self.header_ = None
            X = X.astype(float)
        self.feature_names_ = list(X.columns)
        return X.values, np.asarray(y)

    def _transform(self, X):
        """Apply the thresholds learned in ``fit``."""
        X = self._as_frame(X)
        if not getattr(self, "binarized_", False):
            return X
        X = X[self.input_feature_names_]
        return cut(X.copy(), self.thresholds_)[self.header_].astype(int)

    def fit(self, X_train, y_train, **kwargs):
        """Finds the Rashomon set of falling trees over the training data.
        
        Parameters
        ----------
        X_train : array-like
            The training features.
        y_train : array-like
            The target values.
        **kwargs
            Additional keyword arguments to pass to the GraviTree algorithm.
            Allowed kwargs include:
            - `use_heap`: Whether to use a heap for efficient R-set management (default: True).
            - `max_cache_size`: Maximum size of the cache for intermediate results (default: 1000).
            - `enable_falling_constraint`: Whether to enable the falling constraint (default: True).
            - `budget_override`: Override the default loss + epsilon budget with an absolute budget (default: None).

        When ``binarize`` is true and any feature is not already 0/1, ``fit``
        learns threshold features and ``predict`` / ``score`` apply them.
        ``feature_names_`` then names those binary features, and ``print_tree``
        uses them when ``feature_names`` is omitted.
            
        Returns
        -------
        self : object
            Returns the instance itself.
            
        Sets the `ft_rset` attribute to the resulting R-set from the GraviTree algorithm.
        """
        X_train, y_train = self._prepare_training_data(X_train, y_train)

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp_file:
            dump_path = tmp_file.name
        ft_rset = gravitree_cpp.run_from_xy(X_train,
                                        y_train,
                                        eps=self.eps,
                                        depth=self.depth,
                                        min_support=self.min_support,
                                        branching_cost=self.branching_cost,
                                        dump_rset_jsonl=dump_path,
                                        **kwargs)
        rset = []
        trees1 = open(dump_path).readlines()
        for i in range(ft_rset['rset_size']):
            tree_dict = json.loads(trees1[i])
            rset.append(tree_dict['tree'])
        self.ft_rset = rset
        os.remove(dump_path)
        return self

    def predict_proba(self, X, tree_idx=0):
        """Predicts class probabilities for the input features using the R-set of falling trees.
        
        Parameters
        ----------
        X : array-like or pandas.DataFrame
            The input features for which to predict probabilities.
        tree_idx : int, optional
            The index of the tree in the R-set to use for prediction (default: 0).
            
        Returns
        -------
        proba : numpy.ndarray
            An array of shape (n_samples, n_classes) containing the predicted class probabilities.
        """
        if not hasattr(self, 'ft_rset'):
            raise ValueError("The model has not been fitted yet. Please call fit() before predict_proba().")
        if tree_idx >= len(self.ft_rset) or tree_idx < 0:
            raise ValueError(f"tree_idx {tree_idx} is out of bounds for the R-set size {len(self.ft_rset)}.")
        X = self._transform(X)
        return predict_proba_falling_tree(self.ft_rset[tree_idx], X)

    def predict(self, X, tree_idx=0, threshold=0.5):
        """Predicts class labels for the input features using the R-set of falling trees.
        
        Parameters
        ----------
        X : array-like or pandas.DataFrame
            The input features for which to predict class labels.
        tree_idx : int, optional
            The index of the tree in the R-set to use for prediction (default: 0).
            
        Returns
        -------
        labels : numpy.ndarray
            An array of shape (n_samples,) containing the predicted class labels.
        """
        if not hasattr(self, 'ft_rset'):
            raise ValueError("The model has not been fitted yet. Please call fit() before predict().")
        if tree_idx >= len(self.ft_rset) or tree_idx < 0:
            raise ValueError(f"tree_idx {tree_idx} is out of bounds for the R-set size {len(self.ft_rset)}.")
        X = self._transform(X)
        if isinstance(X, pd.DataFrame):
            X = X.values
        return gravitree_cpp.predict(self.ft_rset[tree_idx], X, threshold=threshold)

    def score(self, X, y, sample_weight=None, tree_idx=0, threshold=0.5):
        """Computes the accuracy of the model on the given test data and labels.
        
        Parameters
        ----------
        X : array-like or pandas.DataFrame
            The input features for which to compute the accuracy.
        y : array-like
            The true class labels for the input features.
        tree_idx : int, optional
            The index of the tree in the R-set to use for prediction (default: 0).
        threshold : float, optional
            The threshold for converting predicted probabilities to class labels (default: 0.5).
        sample_weight : array-like, optional
            Sample weights.
            
        Returns
        -------
        accuracy : float
            The accuracy of the model on the test data.
        """
        if not hasattr(self, 'ft_rset'):
            raise ValueError("The model has not been fitted yet. Please call fit() before score().")
        if tree_idx >= len(self.ft_rset) or tree_idx < 0:
            raise ValueError(f"tree_idx {tree_idx} is out of bounds for the R-set size {len(self.ft_rset)}.")
        preds = self.predict(X, tree_idx=tree_idx, threshold=threshold)
        if isinstance(X, pd.DataFrame):
            X = X.values
        if sample_weight is not None:
            return ((preds == y).astype(float) * sample_weight).sum() / sample_weight.sum()
        return (preds == y).mean()

    def get_branches(self, tree_idx=None):
        """Returns the branches of the specified tree in the R-set.
        
        Parameters
        ----------
        tree_idx : int, optional
            The index of the tree in the R-set for which to return branches (default: None).
            
        Returns
        -------
        branches : list | int
            The number of branches in the specified tree, or a list of branch counts for all trees if tree_idx is None.
        """
        if not hasattr(self, 'ft_rset'):
            raise ValueError("The model has not been fitted yet. Please call fit() before get_branches().")
        if tree_idx is not None and (tree_idx >= len(self.ft_rset) or tree_idx < 0):
            raise ValueError(f"tree_idx {tree_idx} is out of bounds for the R-set size {len(self.ft_rset)}.")
        if tree_idx is None:
            return [gravitree_branches(tree) for tree in self.ft_rset]
        return gravitree_branches(self.ft_rset[tree_idx])

    def get_sparsity(self, tree_idx=None):
        """Returns the sparsity (number of leaves) of the specified tree in the R-set.
        
        Parameters
        ----------
        tree_idx : int, optional
            The index of the tree in the R-set for which to return sparsity (default: None).
            
        Returns
        -------
        sparsity : int or list
            The number of leaves in the specified tree, or a list of leaf counts for all trees if tree_idx is None.
        """
        if not hasattr(self, 'ft_rset'):
            raise ValueError("The model has not been fitted yet. Please call fit() before get_sparsity().")
        if tree_idx is not None and (tree_idx >= len(self.ft_rset) or tree_idx < 0):
            raise ValueError(f"tree_idx {tree_idx} is out of bounds for the R-set size {len(self.ft_rset)}.")
        if tree_idx is None:
            return [gravitree_sparsity(tree) for tree in self.ft_rset]
        return gravitree_sparsity(self.ft_rset[tree_idx])

    def print_tree(self, tree_idx=0, feature_names=None):
        """Prints the structure of the specified tree in the R-set.
        
        Parameters
        ----------
        tree_idx : int, optional
            The index of the tree in the R-set to print (default: 0).
        feature_names : list, optional
            A list of feature names for labeling the tree nodes (default: None).  
        """
        if not hasattr(self, 'ft_rset'):
            raise ValueError("The model has not been fitted yet. Please call fit() before print_tree().")
        if tree_idx >= len(self.ft_rset) or tree_idx < 0:
            raise ValueError(f"tree_idx {tree_idx} is out of bounds for the R-set size {len(self.ft_rset)}.")
        if feature_names is None:
            feature_names = getattr(self, "feature_names_", None)
        if feature_names is not None:
            # If feature names are provided, we can replace feature indices with names in the tree structure
            def replace_feature_names(node):
                if node['t'] == 'L':
                    return node
                else:
                    feature_idx = node['f']
                    node['f'] = feature_names[feature_idx] if feature_idx < len(
                        feature_names) else f"feature_{feature_idx}"
                    node['l'] = replace_feature_names(node['l'])
                    node['r'] = replace_feature_names(node['r'])
                    return node

            tree_to_print = replace_feature_names(copy.deepcopy(self.ft_rset[tree_idx]))
        else:
            tree_to_print = self.ft_rset[tree_idx]

        def print_node(node, indent=""):
            if node['t'] == 'L':
                print(f"{indent}return {node['p']}")
            else:
                print(f"{indent}if {node['f']}:")
                print_node(node['r'], indent + "    ")
                print(f"{indent}else:")
                print_node(node['l'], indent + "    ")

        print_node(tree_to_print)


def visualize_decision_tree(self, tree_idx=0, feature_names=None, output_filename='decision_tree'):
    """
    Generates a visual representation of a decision tree from a dictionary.

    Args:
        tree_dict (dict): The nested dictionary representing the tree.
        feature_names (list): A list of strings where the index corresponds to the 'f' value.
        output_filename (str): The name of the output file (without extension).
    """
    # Initialize the directed graph
    dot = graphviz.Digraph(comment='General Decision Tree', format='png')

    # Global visual settings
    dot.attr(rankdir='TB', size='10,10', dpi='300', nodesep='0.8', ranksep='0.8')
    dot.attr('node', fontname='Helvetica, Arial, sans-serif', fontsize='16')
    dot.attr('edge', fontname='Helvetica, Arial, sans-serif', fontsize='14')

    # Styles
    # decision nodes light grey with dark grey border and black text
    decision_style = {
        'shape': 'box',
        'style': 'filled, rounded',
        'fillcolor': '#F5F5F5',
        'color': '#000000',
        'penwidth': '2'
    }
    # leaf nodes filled with light red/blue based on class, with black text
    # positive class leaf nodes (>=.5 probability) have red border, negative class leaf nodes have blue border
    pos_leaf_style = {
        'shape': 'ellipse',
        'style': 'filled',
        'fillcolor': '#FFEBEE',
        'color': '#F44336',
        'penwidth': '2',
        'fontcolor': '#000000'
    }
    neg_leaf_style = {
        'shape': 'ellipse',
        'style': 'filled',
        'fillcolor': '#E3F2FD',
        'color': '#2196F3',
        'penwidth': '2',
        'fontcolor': '#000000'
    }
    # all edges black with black text
    true_edge = {'color': '#000000', 'fontcolor': '#000000', 'penwidth': '1.5'}
    false_edge = {'color': '#000000', 'fontcolor': '#000000', 'penwidth': '1.5'}

    # Use a mutable list to maintain a unique counter across recursive calls
    node_counter = [0]

    tree_dict = self.ft_rset[tree_idx]

    # Start the recursive build from the root
    build_graph(tree_dict)

    # Render the graph
    dot.render(output_filename, view=False, cleanup=True)
    print(f"Decision tree successfully generated and saved as '{output_filename}_{tree_idx}.png'")


def gravitree_sparsity(tree):
    """Count the number of leaves in a tree returned by GraviTree."""
    if tree['t'] == 'L':
        return 1
    else:
        return gravitree_sparsity(tree['l']) + gravitree_sparsity(tree['r'])


def gravitree_branches(tree):
    """Count the number of nodes with two non-leaf children in a tree returned by GraviTree."""
    if tree['t'] == 'L':
        return 0
    else:
        left_branches = gravitree_branches(tree['l'])
        right_branches = gravitree_branches(tree['r'])
        if tree['l']['t'] != 'L' and tree['r']['t'] != 'L':
            return 1 + left_branches + right_branches
        else:
            return left_branches + right_branches


def build_graph(node, parent_id=None, edge_is_true=None):
    # Generate a unique ID for the current Graphviz node
    current_id = f"node_{node_counter[0]}"
    node_counter[0] += 1

    # 1. Create the node
    if node['t'] == 'L':
        # It's a Leaf
        prob = node['p']
        label = f"Return:\n{prob:.2f}"  # Rounded to 4 decimal places for clarity
        if prob >= 0.5:
            dot.node(current_id, label, **pos_leaf_style)
        else:
            dot.node(current_id, label, **neg_leaf_style)
    elif node['t'] == 'N':
        # It's a Decision Node
        feature_idx = node['f']
        feature_name = feature_names[feature_idx]
        label = f"{feature_name} ?"
        dot.node(current_id, label, **decision_style)

    # 2. Connect to parent (if it exists)
    if parent_id is not None:
        if edge_is_true:
            dot.edge(parent_id, current_id, label=' True', **true_edge)
        else:
            dot.edge(parent_id, current_id, label=' False', **false_edge)

    # 3. Recurse for children (if it's a decision node)
    if node['t'] == 'N':
        # Process Left child (False / Not Satisfied) first so Graphviz renders it on the left
        build_graph(node['l'], parent_id=current_id, edge_is_true=False)

        # Process Right child (True / Satisfied) second so Graphviz renders it on the right
        build_graph(node['r'], parent_id=current_id, edge_is_true=True)
