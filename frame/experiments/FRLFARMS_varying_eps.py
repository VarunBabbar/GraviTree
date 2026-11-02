import argparse
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from treefarms import TREEFARMS
from gosdt import GOSDTClassifier
from gosdt._tree import Node, Leaf
import gosdt
import pickle as pkl
from math import sqrt
import warnings
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description="Run FRL search algorithm.")
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset file')
    parser.add_argument('--curiosity_func', type=str, required=True, help='Curiosity function')
    parser.add_argument('--epsilon', type=float, default=0.02, help='Epsilon')
    parser.add_argument('--reg', type=float, default=REG, help='Regularization')
    parser.add_argument('--filter_antecedents', type=bool, default=False, help='Apply RF filtering')
    return parser.parse_args()


def construct_dataset_from_antecedent_list(df, antecedents):
    df = df.copy()
    columns = df.columns
    selected_columns = []

    for antecedent in antecedents:
        if isinstance(antecedent, int):
            col_name = columns[antecedent]
            selected_columns.append(col_name)
        elif len(antecedent) == 1:
            col_name = columns[antecedent[0]]
            selected_columns.append(col_name)
        else:
            antecedent_name = ' and '.join(columns[i] for i in antecedent)
            if antecedent_name not in df.columns:
                condition = np.ones(df.shape[0], dtype=bool)
                for i in antecedent:
                    condition &= df.iloc[:, i].astype(bool)
                df[antecedent_name] = condition.astype(int)
            selected_columns.append(antecedent_name)

    return df[selected_columns]


def dict_to_tree(tree_dict, X, y, mask=None):
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
    left_child = dict_to_tree(tree_dict["true"], X, y, left_mask)
    right_child = dict_to_tree(tree_dict["false"], X, y, right_mask)
    return Node(feature=feature, left_child=left_child, right_child=right_child)


def _tree_to_dict(node, classes):
    if isinstance(node, gosdt._tree.Leaf):
        return {'prediction': classes[node.prediction]}
    else:
        return {"feature": node.feature,
                "True": _tree_to_dict(node.left_child, classes),
                "False": _tree_to_dict(node.right_child, classes)}


def evaluate_rule(feature, X, mask):
    return (X[:, feature] == 1) & mask

def is_falling_rule_list_tree(tree, X, y):
    mask = np.ones(len(X), dtype=bool)
    probs = []

    def is_leaf(node):
        return isinstance(node, dict) and "prediction" in node

    def check(node, mask):
        if not isinstance(node, dict) or "feature" not in node:
            return False
        feature = node["feature"]
        true_branch = node.get("True")
        false_branch = node.get("False")
        true_mask = evaluate_rule(feature, X, mask)
        false_mask = mask & (~true_mask)

        if not is_leaf(true_branch):
            return False

        y_sub = y[true_mask]
        prob = np.mean(y_sub == 1) if len(y_sub) > 0 else 0.0
        probs.append(prob)

        if is_leaf(false_branch):
            y_sub = y[false_mask]
            prob = np.mean(y_sub == 1) if len(y_sub) > 0 else 0.0
            probs.append(prob)
            return True
        elif "feature" in false_branch:
            return check(false_branch, false_mask)
        else:
            return False

    if not check(tree, mask):
        return False, None, None

    for i in range(len(probs) - 1):
        if probs[i] < probs[i + 1] - 1e-6:
            return False, None, None

    return True, probs, None


def extract_rule_features_in_order(tree):
    features = []
    node = tree
    while 'feature' in node and 'True' in node:
        features.append(node['feature'])
        node = node['False']
    return features

def _num_leaves(tree_as_dict):
    if tree_as_dict is None:
        return -1
    if 'prediction' in tree_as_dict:
        return 1
    return _num_leaves(tree_as_dict['True']) + _num_leaves(tree_as_dict['False'])

def evaluate_treefarms_models_sampled(tf, X, y, config, sample_size=1000000):
    treefarms_frls = []
    objs = []
    matches = 0
    total_trees = tf.get_tree_count()
    X_np = X.values

    # Sample random indices from the Rashomon set
    np.random.seed(42)
    indices = np.random.choice(total_trees, size=min(sample_size, total_trees), replace=False)

    start_time = time.time()
    for j in tqdm(indices):
        model = tf[j]
        tree_dict = vars(model)['source']
        tree = _tree_to_dict(dict_to_tree(tree_dict, X_np, y), [0, 1])
        is_frl, _, _ = is_falling_rule_list_tree(tree, X_np, y)

        if is_frl:
            rule_list = extract_rule_features_in_order(tree)
            if rule_list not in treefarms_frls:
                acc = model.score(X, y)
                num_leaves = _num_leaves(tree)
                obj = acc + config['regularization'] * num_leaves
                treefarms_frls.append(rule_list)
                objs.append(obj)
                matches += 1
    end_time = time.time()
    eval_time = end_time - start_time

    # Estimate the total number of FRLs using the sample proportion
    proportion = matches / len(indices)
    est_total = proportion * total_trees
    if len(indices) < total_trees:
        eval_time_prop = eval_time/len(indices)
        se_eval_time = sqrt(eval_time_prop * (1 - eval_time_prop) / len(indices))*total_trees
        eval_time = eval_time_prop*total_trees
        se_total = sqrt(proportion * (1 - proportion) / len(indices)) * total_trees  # Standard error
    else:
        se_eval_time = 0
        se_total = 0

    return treefarms_frls, objs, eval_time, se_eval_time, est_total, se_total


def count_frls_in_treefarms(epsilon, dataset, reg):
    df = pd.read_csv(f'data/{dataset}.csv')
    X_raw, y = df.iloc[:, :-1].astype(bool), df.iloc[:, -1]

    frl_rset = FRLRashomonSet(epsilon=epsilon, C=0.01)
    frl_rset.fit(X_raw, y, verbose=False, first_pass_iters=20, second_pass_iters=20, curiosity_func="reward")

    antecedents = frl_rset.reference_model.antecedents
    X_used = construct_dataset_from_antecedent_list(X_raw, antecedents)
    config = {
        "regularization": reg,
        "rashomon_bound_adder": epsilon,
        "depth_budget": 5,
        "verbose": False,
        "time_limit": 100
    }

    start_time = time.time()
    tf = TREEFARMS(config)
    tf.fit(X_used, y)
    end_time = time.time()
    train_time = end_time-start_time

    treefarms_frls, _, eval_time, eval_time_std, count, count_std = evaluate_treefarms_models_sampled(tf, X_used, y, config)
    total_time = train_time+eval_time
    total_time_std = eval_time_std
    return len(treefarms_frls), treefarms_frls, total_time, total_time_std, count, count_std


def plot_frl_count_vs_epsilon_grid(epsilon_values, datasets):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    axes2 = axes2.flatten()

    for i, dataset in enumerate(datasets):
        frl_counts = []
        frls = []
        runtimes = []
        frl_counts_std = []
        runtimes_std = []
        REG = 0.01
        for eps in tqdm(epsilon_values, desc=f"Dataset: {dataset}"):
            
            _, treefarms_frls, total_time, total_time_std, count, count_std = count_frls_in_treefarms(eps, dataset,REG)
            frl_counts.append(count)
            frls.append(treefarms_frls)
            runtimes.append(total_time)
            frl_counts_std.append(count_std)
            runtimes_std.append(total_time_std)

        # Save FRLs and runtime info
        with open(f"treefarms_frls_{dataset}.pkl", "wb") as f:
            pkl.dump({
                "dataset": dataset,
                "epsilon_values": epsilon_values,
                "frls_by_epsilon": frls,
                "runtime": runtimes,
                "runtime_std": runtimes_std,
                "counts": frl_counts,
                "count_std": frl_counts_std
            }, f)

        # Plot # Unique FRLs
        ax = axes[i]
        ax.plot(epsilon_values, frl_counts, marker='o', color='tab:blue')
        ax.set_title(f"{dataset}", fontsize=12)
        ax.set_xlabel("Epsilon")
        ax.set_ylabel("# Unique FRLs")
        ax.grid(True)

        # Plot runtime
        ax2 = axes2[i]
        ax2.plot(epsilon_values, runtimes, marker='o', color='tab:red')
        ax2.set_title(f"{dataset}", fontsize=12)
        ax2.set_xlabel("Epsilon")
        ax2.set_ylabel("Runtime (s)")
        ax2.set_yscale('log')
        ax2.grid(True)

    fig.suptitle("TREEFARMS FRLs vs Epsilon", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig("frls_vs_epsilon_treefarms_grid.pdf")

    fig2.suptitle("TREEFARMS Runtime vs Epsilon", fontsize=14)
    fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig2.savefig("runtime_vs_epsilon_treefarms_grid.pdf")


epsilon_values = [0.001,0.002,0.005,0.01,0.015,0.02]
datasets = ["bank", "spambase", "heloc"] # add more datasets here
plot_frl_count_vs_epsilon_grid(epsilon_values, datasets)

