from typing import List, Optional

import networkx as nx
import numpy as np


def check_positive_leaf_falling_constraint(tree, X, y):
    """The positive leaf falling constraint requires that for all paths from root to leaf,
    any positive leaf node adjacent to that path must have a probability below
    any preceding positive leaf node on the path."""
    G = convert_to_positive_leaf_graph(tree, X, y)
    return check_all_paths_descend(G)


def check_full_leaf_falling_constraint(tree, X, y):
    """The full leaf falling constraint requires that for all paths from root to leaf,
    any leaf node adjacent to that path must have a probability below
    any preceding leaf node on the path."""
    G = convert_to_full_leaf_graph(tree, X, y)
    return check_all_paths_descend(G)


def check_depth_based_falling_constraint(tree, X, y):
    """The depth-based falling constraint enforces a recursive falling constraint:
    
    Let n be an internal node of tree T. 
    Let l be the highest probability leaf in the left subtree of n, 
    and r be the highest probability leaf in the right subtree of n.
    Then, the falling constraint requires that:
        - If p_l > p_r, then depth(l) <= depth(r), or
        - If p_r > p_l, then depth(r) <= depth(l)."""
    initialize_leaf_probabilities(tree.source, X, y)

    def check_node(node, current_depth):
        if "prediction" in node:
            return node["probability"], current_depth
        false_child = node["false"]
        true_child = node["true"]
        p_l, d_l = check_node(false_child, current_depth + 1)
        p_r, d_r = check_node(true_child, current_depth + 1)
        if p_l > p_r and d_l > d_r:
            return -1, -1  # Violation
        if p_r > p_l and d_r > d_l:
            return -1, -1  # Violation
        return max(p_l, p_r), min(d_l, d_r)

    max_prob, min_depth = check_node(tree.source, 0)
    return max_prob != -1  # If violation occurred, max_prob would be -1


def check_all_paths_descend(G):
    """Check that for all paths from root to leaf in the graph G, the predicted probabilities descend."""
    for path in nx.all_simple_paths(G, source=0, target=[n for n in G.nodes if G.out_degree(n) == 0]):
        preds = [G.nodes[n]['pred'] for n in path if G.nodes[n]['pred'] >= 0]
        if not all(x >= y for x, y in zip(preds, preds[1:])):
            return False
    return True


def initialize_leaf_probabilities(node, X, y):
    """Recursively initialize the empirical positive proportion at each leaf node."""

    if "prediction" in node:
        node["probability"] = np.mean(y)
        return
    false_child = node["false"]
    true_child = node["true"]
    false_indices = X.iloc[:, node["feature"]] == 0
    true_indices = X.iloc[:, node["feature"]] == 1
    initialize_leaf_probabilities(false_child, X[false_indices], y[false_indices])
    initialize_leaf_probabilities(true_child, X[true_indices], y[true_indices])


def convert_to_positive_leaf_graph(tree, X, y):
    initialize_leaf_probabilities(tree.source, X, y)
    tree_source = tree.source
    G = nx.DiGraph()
    node_queue = [tree_source]
    parents: List[Optional[int]] = [None]
    node_counter = 0
    while node_queue:
        node = node_queue.pop(0)
        if "prediction" in node:
            continue
        false_child = node["false"]
        true_child = node["true"]
        false_leaf = "prediction" in false_child
        true_leaf = "prediction" in true_child

        # if either child is a positive leaf, add it to the graph with its probability
        if false_leaf and false_child["prediction"] == 1:
            G.add_node(node_counter, pred=false_child["probability"], feature=node["feature"])
        elif true_leaf and true_child["prediction"] == 1:
            G.add_node(node_counter, pred=true_child["probability"], feature=node["feature"])
        # if neither child is a positive leaf, add the internal node as a dummy split
        else:
            G.add_node(node_counter, pred=-1, feature=node["feature"])

        if parents[0] is not None:
            G.add_edge(parents[0], node_counter)

        # keep track of parents and continue traversal
        parents.pop(0)
        node_counter += 1
        if not false_leaf:
            node_queue.insert(0, false_child)
            parents.insert(0, node_counter - 1)
        if not true_leaf:
            node_queue.insert(0, true_child)
            parents.insert(0, node_counter - 1)
    return G


def convert_to_full_leaf_graph(tree, X, y):
    initialize_leaf_probabilities(tree.source, X, y)
    tree_source = tree.source
    G = nx.DiGraph()
    node_queue = [tree_source]
    parents: List[Optional[int]] = [None]
    node_counter = 0
    while node_queue:
        node = node_queue.pop(0)
        if "prediction" in node:
            continue
        false_child = node["false"]
        true_child = node["true"]
        false_leaf = "prediction" in false_child
        true_leaf = "prediction" in true_child
        current_node = node_counter

        # if both children are leaves, add them both to the graph with their probabilities beneath a dummy split
        if false_leaf and true_leaf:
            G.add_node(node_counter, pred=-1, feature=node["feature"])
            G.add_node(node_counter + 1, pred=false_child["probability"], feature=node["feature"])
            G.add_node(node_counter + 2, pred=true_child["probability"], feature=node["feature"])
            G.add_edge(node_counter, node_counter + 1)
            G.add_edge(node_counter, node_counter + 2)
            node_counter += 2
        # if only one child is a leaf, add that child with its probability
        elif false_leaf:
            G.add_node(node_counter, pred=false_child["probability"], feature=node["feature"])
        elif true_leaf:
            G.add_node(node_counter, pred=true_child["probability"], feature=node["feature"])
        # if neither child is a leaf, add a dummy split
        else:
            G.add_node(node_counter, pred=-1, feature=node["feature"])

        if parents[0] is not None:
            G.add_edge(parents[0], current_node)

        # keep track of parents and continue traversal
        parents.pop(0)
        node_counter += 1
        if not false_leaf:
            node_queue.insert(0, false_child)
            parents.insert(0, current_node)
        if not true_leaf:
            node_queue.insert(0, true_child)
            parents.insert(0, current_node)
    return G
