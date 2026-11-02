import numpy as np
import pandas as pd
from tqdm import tqdm
from treefarms import TREEFARMS
from gosdt._tree import Node, Leaf
    
class FRLFarms:
    def __init__(self, epsilon=0.01, regularization=0.01, max_len=1, min_support=0.01, max_sample_limit=100000):
        self.epsilon = epsilon
        self.regularization = regularization
        self.max_len = max_len
        self.min_support = min_support
        self.max_sample_limit = max_sample_limit
        self.rashomon_frls = []
        self.objective_values = []

    def _construct_dataset(self, X, antecedents):
        df = X.copy()
        columns = df.columns
        selected_columns = []

        for antecedent in antecedents:
            if isinstance(antecedent, int):
                selected_columns.append(columns[antecedent])
            elif len(antecedent) == 1:
                selected_columns.append(columns[antecedent[0]])
            else:
                name = ' and '.join(columns[i] for i in antecedent)
                if name not in df.columns:
                    condition = np.ones(df.shape[0], dtype=bool)
                    for i in antecedent:
                        condition &= df.iloc[:, i].astype(bool)
                    df[name] = condition.astype(int)
                selected_columns.append(name)

        return df[selected_columns]

    def dict_to_tree(self, tree_dict, X, y, mask=None):
        if mask is None:
            mask = np.ones(len(X), dtype=bool)
    
        if "prediction" in tree_dict:
            pred = int(tree_dict["prediction"])
            loss = (y[mask] != pred).sum() / mask.sum()
            return Leaf(prediction=pred, loss=loss)
    
        feature = tree_dict["feature"]
        feature_vals = X[:, feature]
        left_mask = mask & (feature_vals == 1)
        right_mask = mask & (feature_vals == 0)
        left_child = self.dict_to_tree(tree_dict["true"], X, y, left_mask)
        right_child = self.dict_to_tree(tree_dict["false"], X, y, right_mask)
        return Node(feature=feature, left_child=left_child, right_child=right_child)
        
    def _evaluate_tree(self, tree, X, y):
        mask = np.ones(len(X), dtype=bool)
        probs = []

        def is_leaf(node):
            return isinstance(node, dict) and "prediction" in node

        def check(node, mask):
            if not isinstance(node, dict) or "feature" not in node:
                return False
            feature = node["feature"]
            true_mask = (X[:, feature] == 1) & mask
            false_mask = mask & (~true_mask)

            true_branch = node.get("True")
            false_branch = node.get("False")

            if not is_leaf(true_branch):
                return False

            probs.append(np.mean(y[true_mask] == 1) if len(y[true_mask]) > 0 else 0.0)

            if is_leaf(false_branch):
                probs.append(np.mean(y[false_mask] == 1) if len(y[false_mask]) > 0 else 0.0)
                return True
            else:
                return check(false_branch, false_mask)

        if not check(tree, mask):
            return False, None

        for i in range(len(probs) - 1):
            if probs[i] < probs[i + 1] - 1e-6:
                return False, None

        return True, probs

    def _extract_features(self, tree):
        features = []
        node = tree
        while 'feature' in node and 'True' in node:
            features.append(node['feature'])
            node = node['False']
        return features

    def _count_leaves(self, tree):
        if 'prediction' in tree:
            return 1
        return self._count_leaves(tree['True']) + self._count_leaves(tree['False'])

    def _tree_to_dict(self, node, classes):
        if isinstance(node, gosdt._tree.Leaf):
            return {'prediction': classes[node.prediction]}
        else:
            return {"feature": node.feature,
                    "True": self._tree_to_dict(node.left_child, classes),
                    "False": self._tree_to_dict(node.right_child, classes)}

    def fit(self, X, y):
        X_bin = X.astype(bool)

        config = {
            "regularization": self.regularization,
            "rashomon_bound_adder": self.epsilon,
            "depth_budget": 5,
            "verbose": False,
            "time_limit": 100
        }

        tf = TREEFARMS(config)
        tf.fit(X_bin, y)

        total_trees = tf.get_tree_count()
        max_sample_limit = min(self.max_sample_limit, total_trees)
        indices = np.random.choice(total_trees, size=max_sample_limit, replace=False)
        X_np = X_bin.values

        for i in tqdm(indices, desc="Evaluating TREEFARMS trees"):
            model = tf[i]
            tree_dict = vars(model)['source']
            tree = self._tree_to_dict(self.dict_to_tree(tree_dict, X_np, y), [0, 1])
            is_frl, _ = self._evaluate_tree(tree, X_np, y)
            if is_frl:
                rules = self._extract_features(tree)
                if rules not in self.rashomon_frls:
                    acc = model.score(X_bin, y)
                    obj = acc + self.regularization * self._count_leaves(tree)
                    self.rashomon_frls.append(rules)
                    self.objective_values.append(obj)

    def get_frls(self):
        return self.rashomon_frls, self.objective_values
