import numpy as np
from frame.bounds import improvement_is_possible, antecedent_is_feasible, objective_lower_bound_addition
from frame.data_precomputation import index_list_to_bitmask
from tqdm import tqdm
from gmpy2 import popcount
from frame.Trie import FallingRuleListTrie
from frame.curiosity_functions import ucb_curiosity

DEFAULT_LAMBDA = 0.8
DEFAULT_TERMINATION_PROBABILITY = 0.01
DEFAULT_MAX_ITERS = 10000


def optimize_falling_rule_list(X,
                               y,
                               antecedents,
                               antecedent_map,
                               w,
                               C,
                               opt_curiosity="paper",
                               max_iters=DEFAULT_MAX_ITERS,
                               terminate_prob=DEFAULT_TERMINATION_PROBABILITY,
                               verbose=False):
    """Optimize a falling rule list using Algorithm 1 from the paper.

    Parameters
    ----------
    X : pd.DataFrame | np.ndarray
        The dataset.
    y : pd.Series | np.ndarray
        The labels.
    antecedents : list[tuple]
        A list of antecedents.
    antecedent_map : dict
        A map from antecedents to the indices of the points that satisfy them.
    w : float
        The positive class weight.
    C : float
        The penalty for adding a new rule.
    max_iters : int, optional
        The maximum number of iterations.
    terminate_prob : float, optional
        The probability of terminating the optimization.
    verbose : bool, optional
        Whether to display a progress bar.

    Returns
    -------
    list[tuple]
        The optimized rule list.
    """
    n = X.shape[0]
    best_rule_list = []
    best_obj = np.inf

    trie = FallingRuleListTrie()

    # bitmasks for the indices of the positive and negative instances
    base_pos = index_list_to_bitmask(np.where(y == 1)[0], n)
    base_neg = index_list_to_bitmask(np.where(y == 0)[0], n)

    for i in tqdm(range(max_iters), "Optimizing rule list", disable=(not verbose)):
        prefix = []
        prefix_obj = 0
        alpha = 1

        # bitmasks for the remaining positive and negative instances after the current prefix
        pos_after_prefix = base_pos
        neg_after_prefix = base_neg
        n_pos_after_prefix = pos_after_prefix.bit_count()
        n_neg_after_prefix = neg_after_prefix.bit_count()

        while improvement_is_possible(n, n_pos_after_prefix, n_neg_after_prefix, alpha, w, C):
            if np.random.rand() < terminate_prob:
                break

            candidate_antecedents = []
            curiosities = []

            for antecedent in antecedents:
                # find all the instances captured by the antecedent
                pos_captured_full_data, neg_captured_full_data = antecedent_map[antecedent]['pos'], antecedent_map[
                    antecedent]['neg']

                # find the instances captured by the antecedent w.r.t the current prefix
                pos_captured, neg_captured, n_pos_captured, n_neg_captured = captured_instances(
                    pos_captured_full_data, neg_captured_full_data, pos_after_prefix, neg_after_prefix)

                n_captured = n_pos_captured + n_neg_captured

                # find the instances left uncaptured by the antecedent and the rest of the prefix
                _, _, n_pos_after_new_rule, n_neg_after_new_rule = remaining_instances(
                    pos_captured, neg_captured, pos_after_prefix, neg_after_prefix)
                n_after_new_rule = n_pos_after_new_rule + n_neg_after_new_rule

                if n_captured == 0 or n_after_new_rule == 0:
                    continue

                antecedent_alpha = n_pos_captured / (n_captured)

                if antecedent_is_feasible(n_neg_after_new_rule, n_pos_after_new_rule, antecedent_alpha, alpha, w):
                    loss_of_new_rule = _loss_due_to_rule(n_pos_captured, n_neg_captured, w, n)
                    bound = objective_lower_bound_addition(n, n_pos_after_new_rule, n_neg_after_new_rule, alpha, w, C)
                    if prefix_obj + loss_of_new_rule + C + bound < best_obj:
                        candidate_antecedents.append(antecedent)
                        tnode = trie.find_node(prefix, antecedent)
                        visits = tnode.visits if tnode else 0
                        if opt_curiosity == "paper":
                            curiosities.append(_paper_curiosity(alpha, n_pos_captured, n_pos_after_new_rule))
                        elif opt_curiosity == "ucb":
                            curiosities.append(ucb_curiosity(alpha, visits, i))

            if not candidate_antecedents:
                break

            # choose the next antecedent and update the prefix and counts
            next_antecedent = _choose_next_antecedent(candidate_antecedents, curiosities)

            # find all the instances captured by the new antecedent
            pos_captured_full_data, neg_captured_full_data = antecedent_map[next_antecedent]['pos'], antecedent_map[
                next_antecedent]['neg']

            # find the instances captured by the new antecedent
            pos_captured, neg_captured, n_pos_captured, n_neg_captured = captured_instances(
                pos_captured_full_data, neg_captured_full_data, pos_after_prefix, neg_after_prefix)

            # find the instances left uncaptured by the new antecedent and the rest of the prefix
            pos_after_prefix, neg_after_prefix, n_pos_after_prefix, n_neg_after_prefix = remaining_instances(
                pos_captured, neg_captured, pos_after_prefix, neg_after_prefix)

            # add the new antecedent to the prefix and update the objective
            n_captured = n_pos_captured + n_neg_captured
            alpha = n_pos_captured / n_captured

            prefix_obj += _loss_due_to_rule(n_pos_captured, n_neg_captured, w, n) + C
            prefix.append((next_antecedent, alpha))

        # calculate the objective including the remaining instances after the prefix
        loss_remaining = _loss_due_to_rule(n_pos_after_prefix, n_neg_after_prefix, w, n)
        obj = prefix_obj + loss_remaining

        prefix.append(((), n_pos_after_prefix / (n_pos_after_prefix + n_neg_after_prefix)))
        trie.insert(prefix, obj)

        if obj < best_obj:
            best_obj = obj
            best_rule_list = prefix

    return best_rule_list, best_obj


def _choose_next_antecedent(feasible_antecedents: list[tuple], curiosities: list[float]) -> tuple:
    """Choose the next antecedent to add to the prefix.
    
    Parameters
    ----------
    feasible_antecedents : list[tuple]
        A list of antecedents that can be added to the prefix.
    curiosities : list[float]
        A list of curiosities corresponding to the antecedents.
    
    Returns
    -------
    tuple
        The antecedent to add to the prefix.
    """
    norm = np.sum(curiosities)
    if norm == 0:
        feasible_sampling_probabilities = np.ones(len(feasible_antecedents)) / len(feasible_antecedents)
    else:
        feasible_sampling_probabilities = curiosities / norm
    random_idx = np.random.choice(len(feasible_antecedents), p=feasible_sampling_probabilities)
    return feasible_antecedents[random_idx]


def _paper_curiosity(alpha_rule: float, n_pos_captured_by_rule: int, n_pos_after: int, lmbda=DEFAULT_LAMBDA):
    """Compute the curiosity of a rule.
    
    Parameters
    ----------
    alpha_rule : float
        The empirical positive proportion of the rule.
    n_pos_captured_by_rule : int
        The number of positive instances captured by the rule.
    n_pos_after : int
        The number of positive instances after the rule.
    lmbda : float, optional
        The tradeoff between emp pos prop and ratio of remaining pos instances captured by the rule.

    Returns
    -------
    float
        The curiosity of the rule.
    """
    curiosity = (lmbda * alpha_rule) + ((1 - lmbda) * (n_pos_captured_by_rule / (n_pos_after + 1)))
    return curiosity


def _loss_due_to_rule(n_pos_captured_by_rule: int, n_neg_captured_by_rule: int, w: float, n: int):
    """Compute the loss due solely to the rule.
    
    Parameters
    ----------
    n_pos_captured_by_rule : int
        The number of positive instances captured by the rule.
    n_neg_captured_by_rule : int
        The number of negative instances captured by the rule.
    w : float
        The positive class weight.
    n : int
        The total number of instances.

    Returns
    -------
    float
        The loss due to the rule.
    """
    if (w * n_pos_captured_by_rule > n_neg_captured_by_rule):
        return (1 / n) * n_neg_captured_by_rule
    else:
        return (w / n) * n_pos_captured_by_rule


def captured_instances(pos_captured_full_data, neg_captured_full_data, remaining_pos: int, remaining_neg: int):
    """Find the instances captured by the rule.
    
    This operation is done by an intersection of the bitmasks of the
    instances captured by the rule and the remaining instances.
    
    Parameters
    ----------
    rule : tuple
        The rule.
    antecedent_map : dict
        The map from antecedents to captured instances.
    remaining_pos : int
        The bitmask of the remaining positive instances.
    remaining_neg : int
        The bitmask of the remaining negative instances.

    Returns
    -------
    int
        The bitmask of the positive instances captured by the rule.
    int
        The bitmask of the negative instances captured by the rule.
    """
    pos_captured_by_rule = pos_captured_full_data & remaining_pos
    neg_captured_by_rule = neg_captured_full_data & remaining_neg

    return pos_captured_by_rule, neg_captured_by_rule, popcount(pos_captured_by_rule), popcount(neg_captured_by_rule)


def remaining_instances(pos_captured_by_rule: int, neg_captured_by_rule: int, remaining_pos: int, remaining_neg: int):
    """Find the uncaptured instances which remain uncaptured after the rule.

    This operation is done by a difference of the bitmasks of the
    instances captured by the rule and the remaining instances.
    
    Parameters
    ----------
    pos_captured_by_rule : int
        The bitmask of the positive instances captured by the rule.
    neg_captured_by_rule : int
        The bitmask of the negative instances captured by the rule.
    remaining_pos : int
        The bitmask of the remaining positive instances.
    remaining_neg : int
        The bitmask of the remaining negative instances.

    Returns
    -------
    int
        The bitmask of the positive instances remaining uncaptured after the rule.
    int
        The bitmask of the negative instances remaining uncaptured after the rule.
    """

    remaining_pos_after_rule = remaining_pos & ~pos_captured_by_rule
    remaining_neg_after_rule = remaining_neg & ~neg_captured_by_rule

    return remaining_pos_after_rule, remaining_neg_after_rule, popcount(remaining_pos_after_rule), popcount(
        remaining_neg_after_rule)
