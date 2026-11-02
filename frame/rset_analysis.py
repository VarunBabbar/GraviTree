import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from frame.rashomon_sets import FRLRashomonSet
import matplotlib.pyplot as plt
from tqdm import tqdm
import re


def rashomon_set_stats(rset, X, y):
    """
    Print statistics about a Falling Rule List Rashomon Set.
    
    Parameters
    ----------
    rset : rashomon_sets.FRLRashomonSet
        The Rashomon set object to analyze
    X : pd.DataFrame | np.ndarray
        The dataset used to fit the model
    y : pd.Series | np.ndarray
        The labels for the dataset
        
    Returns
    -------
    None
        Prints statistics to output
    """
    total_models = len(rset.rset)
    unique_models = len(set(rset.rset))
    ref_obj = rset.reference_model.objective(X, y)
    rset_objs = [frl.objective(X, y) for frl in rset.rset]
    min_obj = min(rset_objs)
    max_obj = max(rset_objs)

    print(f'Length of the rashomon set with duplicates: {total_models}')
    print(f'Length of the rashomon set without duplicates: {unique_models}')
    print(f'Reference FRL objective: {ref_obj}')
    print(f'Minimum objective in Rashomon set: {min_obj}')
    print(f'Maximum objective in Rashomon set: {max_obj}')


def antecedent_to_label(antecedent, feature_names):
    """
    Convert antecedent indices to a readable feature label string.
    
    Parameters
    ----------
    antecedent : tuple
        A tuple of indices representing features in an antecedent
    feature_names : list-like
        The list of feature names corresponding to the indices

    Returns
    -------
    str
        A string representation of the antecedent using feature names
    """
    if not antecedent:
        return "No Antecedent"
    return " & ".join([feature_names[i] for i in antecedent])


def antecedent_usage(rset):
    """
    Compute the usage of antecedents across a unique Rashomon set of Falling Rule Lists.

    Parameters
    ----------
    rset : rashomon_sets.FRLRashomonSet
        The Rashomon set object to analyze

    Returns
    -------
    pd.DataFrame
        A DataFrame with antecedents, their count, and usage statistics.
    """
    unique_rset = list(set(rset.rset))
    num_models = len(unique_rset)
    feature_names = list(rset.reference_model.features)

    all_rules = []
    for frl in unique_rset:
        all_rules.extend([tuple(rule[0]) for rule in frl.rule_list if rule[0]])

    rule_counts = Counter(all_rules)
    rule_usage = {rule: count / num_models for rule, count in rule_counts.items()}

    usage_df = pd.DataFrame({
        'Antecedent': [antecedent_to_label(rule, feature_names) for rule in rule_usage.keys()],
        'Count': [rule_counts[rule] for rule in rule_usage.keys()],
        'Usage': list(rule_usage.values())
    })

    usage_df = usage_df.sort_values('Usage', ascending=False).reset_index(drop=True)
    return usage_df


def levenshtein_distance(frl1, frl2):
    """
    Compute the Levenshtein Distance between two Falling Rule List models.
    Each antecedent counts as one unit of difference.

    Parameters
    ----------
    frl1 : FallingRuleList
        First model to compare.
    frl2 : FallingRuleList
        Second model to compare.
        
    Returns
    -------
    int
        The Levenshtein Distance between the two rule lists.
    """
    # Extract the antecedents from each model
    rules1 = [tuple(rule[0]) for rule in frl1.rule_list if rule[0]]
    rules2 = [tuple(rule[0]) for rule in frl2.rule_list if rule[0]]

    # Initialize distance matrix
    len1, len2 = len(rules1), len(rules2)
    dp = np.zeros((len1 + 1, len2 + 1), dtype=int)

    # Base cases: cost of transforming empty sequence
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    # Compute distance
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if rules1[i - 1] == rules2[j - 1]:  # No change needed
                cost = 0
            else:  # Replace operation
                cost = 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # Deletion
                dp[i][j - 1] + 1,  # Insertion
                dp[i - 1][j - 1] + cost  # Substitution
            )

    return dp[len1][len2]


def prediction_distance(frl1, frl2, X):
    """
    Compute the number of differing predictions between two Falling Rule List models.

    Parameters
    ----------
    frl1 : FallingRuleList
        First model to compare.
    frl2 : FallingRuleList
        Second model to compare.

    Returns:
    int
        The number of instances where the predictions differ.
    """
    # Get predictions from both models
    preds1 = frl1.predict(X)
    preds2 = frl2.predict(X)

    # Calculate the number of differing predictions
    pred_distance = np.sum(preds1 != preds2)

    return pred_distance


def _extract_base_feature_name(feature_condition_string):
    """
    Extracts the base feature name from a feature condition string.
    Example: '~Gang_Affiliated<=0.5' -> 'Gang_Affiliated'
             'Percent_Days_Employed<=0.5313' -> 'Percent_Days_Employed'
    """
    if feature_condition_string.startswith('~'):
        name_part = feature_condition_string[1:]
    else:
        name_part = feature_condition_string
    match = re.match(r"([^<>=]+)", name_part)
    if match:
        return match.group(1)
    return name_part


def _get_model_base_feature_set(frl_model, all_feature_conditions_list):
    """
    Extracts the set of unique base feature names used by an FRL model.
    ... (docstring unchanged)
    """
    used_base_features = set()
    if not hasattr(frl_model, 'rule_list'):
        print(f"Warning: Model {frl_model} does not have 'rule_list' attribute.")
        return used_base_features

    antecedent_indices_tuples = [rule[0] for rule in frl_model.rule_list if rule[0]]
    for indices_tuple in antecedent_indices_tuples:
        for index in indices_tuple:
            if 0 <= index < len(all_feature_conditions_list):
                feature_condition = all_feature_conditions_list[index]
                base_name = _extract_base_feature_name(feature_condition)
                used_base_features.add(base_name)
            else:
                print(f"Warning: Antecedent index {index} is out of bounds "
                      f"for the provided feature conditions list (length {len(all_feature_conditions_list)}).")
    return used_base_features


def feature_set_hamming_distance(frl1, frl2, model_feature_conditions_list):
    """
    Compute the Hamming Distance between the base feature sets of two FRL models.
    The distance is the number of unique base features that are used in one model
    but not the other.

    Parameters
    ----------
    frl1 : object
        First FRL model. Must have a `rule_list` attribute.
    frl2 : object
        Second FRL model, with the same structure as frl1.
    model_feature_conditions_list : list of str
        The authoritative list of feature condition strings that the indices
        in the FRL models refer to (e.g., from `list(rset.reference_model.features)`).

    Returns
    -------
    int
        The Hamming Distance between the sets of base features used by the two models.
    """
    if not isinstance(model_feature_conditions_list, list):
        raise TypeError("model_feature_conditions_list must be a list.")
    if not all(isinstance(item, str) for item in model_feature_conditions_list):
        raise ValueError("All items in model_feature_conditions_list must be strings.")

    base_features_model1 = _get_model_base_feature_set(frl1, model_feature_conditions_list)
    base_features_model2 = _get_model_base_feature_set(frl2, model_feature_conditions_list)

    distance = len(base_features_model1.symmetric_difference(base_features_model2))
    return distance


def calc_distance_matrix(unique_rset, distance_metric='levenshtein', X=None, model_feature_lookup_list=None):
    """
    Compute a symmetric pairwise distance matrix for unique models.

    Parameters
    ----------
    unique_rset : list
        A list of unique FRL models.
    distance_metric : str, optional
        The distance metric to use. Options are:
        - 'levenshtein': Computes rule list differences.
        - 'prediction': Computes differences in predictions (requires `X`).
        - 'feature_set_hamming': Computes Hamming distance between base feature sets
                                 (requires `model_feature_lookup_list`).
        Defaults to 'levenshtein'.
    X : pd.DataFrame or np.ndarray, optional
        The dataset. Required ONLY if `distance_metric` is 'prediction'.
    model_feature_lookup_list : list of str, optional
        The authoritative list of feature condition strings that model indices refer to.
        Required ONLY if `distance_metric` is 'feature_set_hamming'.
        E.g., `list(rset.reference_model.features)`.

    Returns
    -------
    np.ndarray
        A square symmetric matrix with pairwise distances between the models.
    """
    supported_metrics = ['levenshtein', 'prediction', 'feature_set_hamming']
    if distance_metric not in supported_metrics:
        raise ValueError(f"Distance metric '{distance_metric}' is not supported. "
                         f"Supported metrics are: {supported_metrics}")

    if distance_metric == 'prediction' and X is None:
        raise ValueError("Dataset X must be provided for 'prediction' distance calculation.")

    if distance_metric == 'feature_set_hamming':
        if model_feature_lookup_list is None:
            raise ValueError("'model_feature_lookup_list' must be provided for 'feature_set_hamming' distance.")

    num_models = len(unique_rset)
    distance_matrix = np.zeros((num_models, num_models))
    total_iterations = (num_models * (num_models - 1)) // 2 if num_models > 0 else 0

    #with tqdm(total=total_iterations, desc=f"Computing {distance_metric} distances") as pbar:
    for i in range(num_models):
        for j in range(i + 1, num_models):
            model1 = unique_rset[i]
            model2 = unique_rset[j]
            d = 0

            if distance_metric == 'levenshtein':
                d = levenshtein_distance(model1, model2)
            elif distance_metric == 'prediction':
                d = prediction_distance(model1, model2, X)
            elif distance_metric == 'feature_set_hamming':
                d = feature_set_hamming_distance(model1, model2, model_feature_lookup_list)

            distance_matrix[i, j] = d
            distance_matrix[j, i] = d
            #pbar.update(1)

    return distance_matrix


def plot_model_differences(distance_matrix):
    """
    Plot a given distance matrix

    Parameters
    ----------
    distance_matrix : np.ndarray
        A square matrix of pairwise distances between models.

    Returns
    -------
    None
        Plots distance matrix
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(distance_matrix, cmap='Blues', interpolation='nearest')
    plt.colorbar(label="Distance")
    plt.title("Distance Matrix")
    plt.xlabel("Model Index")
    plt.ylabel("Model Index")
    plt.show()


def max_difference(distance_matrix):
    """
    Return the indices of the model pair with the maximum distance.

    Parameters
    ----------
    distance_matrix : np.ndarray
        A square matrix containing pairwise distances between models.

    Returns
    -------
    tuple of int
        The indices (i, j) of the two models with the maximum distance.
    """
    max_index = np.unravel_index(np.argmax(distance_matrix, axis=None), distance_matrix.shape)
    return max_index


def average_distance(distance_matrix):
    """
    Compute the average pairwise distance, excluding diagonal elements.

    Parameters
    ----------
    distance_matrix : np.ndarray
        A square matrix containing pairwise distances between models.

    Returns
    -------
    float
        The average distance computed from all off-diagonal elements.
    """
    n = distance_matrix.shape[0]
    total = np.sum(distance_matrix)
    count = n * (n - 1)
    return total / count if count > 0 else 0


def find_model_groups_by_distance(unique_rset, distance_matrix, k=0):
    """
    Identify groups of models in which each model's prediction distance to a reference model is at most k.
    Note, the reference model is chosen randomly within each group. 

    Parameters
    ----------
    unique_rset : list[FallingRuleList]
        A list of unique Falling Rule List models (duplicates not allowed).
    distance_matrix : np.ndarray
        A square matrix of pairwise prediction distances between models.
    k : int, optional
        Maximum allowed prediction distance for grouping. Default is 0.

    Returns
    -------
    list[set[int]]
        A list of sets, where each set contains the indices (as ints) of models in the same group.
    """
    num_models = len(unique_rset)
    groups = []
    visited = set()

    for i in range(num_models):
        if i in visited:
            continue

        # Use vectorized comparison and convert each index to an int
        group = {int(j) for j in np.where(distance_matrix[i, :] <= k)[0]}
        group.add(i)  # Ensure the reference model is included

        groups.append(group)
        visited.update(group)

    # Print group summary
    print(f"Number of groups found with pred distance ≤ {k}: {len(groups)}")
    for idx, group in enumerate(groups, start=1):
        print(f"Group {idx}: {sorted(group)}")

    return groups


def plot_pred_vs_edit_distance(edit_distance_matrix, pred_distance_matrix):
    """
    Plot the average prediction distance as a function of edit distance.

    Parameters
    ----------
    edit_distance_matrix : np.ndarray
        A symmetric matrix containing pairwise edit distances between models.
    pred_distance_matrix : np.ndarray
        A symmetric matrix containing pairwise prediction distances between models.

    Returns
    -------
    None
        Displays a plot of average prediction distance versus edit distance.
    """

    num_models = edit_distance_matrix.shape[0]
    edit_distance_groups = defaultdict(list)

    # Collect prediction distances for each edit distance (only unique pairs)
    for i in range(num_models):
        for j in range(i + 1, num_models):
            edit_dist = edit_distance_matrix[i, j]
            pred_dist = pred_distance_matrix[i, j]
            edit_distance_groups[edit_dist].append(pred_dist)

    # Compute average prediction distance for each unique edit distance
    edit_distance_values = sorted(edit_distance_groups.keys())
    avg_pred_distances = [sum(edit_distance_groups[ed]) / len(edit_distance_groups[ed]) for ed in edit_distance_values]

    # Plot the results
    plt.figure(figsize=(8, 6))
    plt.plot(edit_distance_values, avg_pred_distances, marker='o', linestyle='-')
    plt.xlabel("Edit Distance")
    plt.ylabel("Average Prediction Distance")
    plt.title("Average Prediction Distance vs Edit Distance")
    plt.grid(True)
    plt.show()


def calculate_variable_importance(rset, X, y, b=10):
    """
    Calculate variable importance for each feature in X over the Rashomon set.

    Parameters:
    rset: FRLRashomonSet
        The Rashomon set containing multiple Falling Rule List models.
    X: pd.DataFrame
        The feature dataset.
    y: pd.Series or np.ndarray
        The target labels.
    b: int, optional (default=10)
        The number of bootstrap samples to take for each variable.

    Returns:
    dict:
        A dictionary containing variable names as keys and their corresponding 
        bootstrapped objective increase errors as values.
    """
    unique_rset = list(set(rset.rset))  # Get unique models in the Rashomon set
    num_models = len(unique_rset)
    feature_names = X.columns
    variable_importance = {feature: [] for feature in feature_names}

    for frl in tqdm(unique_rset):
        original_obj = frl.objective(X, y)

        for feature in feature_names:
            bootstrap_obj_diffs = []

            for _ in range(b):
                # Bootstrap the feature column by shuffling it
                X_bootstrapped = X.copy()
                X_bootstrapped[feature] = np.random.permutation(X_bootstrapped[feature].values)

                # Compute objective on bootstrapped data
                bootstrapped_obj = frl.objective(X_bootstrapped, y)

                # Store the increase in objective
                bootstrap_obj_diffs.append(bootstrapped_obj - original_obj)

            # Average over bootstrap runs and store the result
            variable_importance[feature].append(np.mean(bootstrap_obj_diffs))

    # Average over all models in the Rashomon set
    variable_importance_avg = {feature: np.mean(variable_importance[feature]) for feature in feature_names}

    return variable_importance_avg
