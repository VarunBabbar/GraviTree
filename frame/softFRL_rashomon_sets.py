import numpy as np
from tqdm import tqdm
import time

from frame.optimization import DEFAULT_MAX_ITERS, DEFAULT_TERMINATION_PROBABILITY
from frame.optimization import captured_instances, remaining_instances, _choose_next_antecedent, _softFRL_loss_due_to_rule, softfrl_objective_lower_bound_addition
from frame.data_precomputation import index_list_to_bitmask
from FRL import DEFAULT_POSITIVE_WEIGHT, DEFAULT_RULE_PENALTY
from frame.SoftFRL import SoftFRL, DEFAULT_FALLING_PENALTY
from frame.Trie import TrieNode, FallingRuleListTrie
from frame.curiosity_functions import *

DEFAULT_EPSILON = 1e-2


class SoftFRLRashomonSet:

    def __init__(self,
                 w=DEFAULT_POSITIVE_WEIGHT,
                 C=DEFAULT_RULE_PENALTY,
                 C1=DEFAULT_FALLING_PENALTY,
                 epsilon=DEFAULT_EPSILON):
        self.w = w
        self.C = C
        self.C1 = C1
        self.epsilon = epsilon
        self.unique_model_counts = []  # list to track size of rset over iterations

    def fit(self,
            X,
            y,
            first_pass_iters=DEFAULT_MAX_ITERS,
            second_pass_iters=DEFAULT_MAX_ITERS,
            curiosity_func='paper',
            best_obj=None,
            **kwargs):
        self.reference_model = SoftFRL(w=self.w, C=self.C, C1=self.C1)
        self.reference_model.fit(X, y, save_precomputation=True, max_iters=first_pass_iters, **kwargs)
        if best_obj is None:
            best_obj = self.reference_model.objective(X, y)

        start_time = time.time()  # timing for second pass of algorithm
        rule_lists, unique_counts, unique_models = soft_falling_rule_list_rset(X,
                                                                               y,
                                                                               self.reference_model.antecedents,
                                                                               self.reference_model.antecedent_map,
                                                                               self.w,
                                                                               self.C,
                                                                               self.C1,
                                                                               best_obj,
                                                                               self.epsilon,
                                                                               curiosity_func=curiosity_func,
                                                                               max_iters=second_pass_iters,
                                                                               **kwargs)
        end_time = time.time()
        self.second_pass_time = end_time - start_time
        self.rset = [SoftFRL.from_rule_list(rule_list, w=self.w, C=self.C, C1=self.C1) for rule_list in rule_lists]
        self.unique_model_counts = unique_counts
        self.unique_models = unique_models

        # initialize features, pos_list, neg_list for all FRLs in the Rashomon set
        for frl in self.rset:
            frl.features = self.reference_model.features
            if self.reference_model.included_complement:
                X_ = X.copy()
                for col in X.columns:
                    X_['~' + col] = ~X[col]
            else:
                X_ = X
            frl.included_complement = self.reference_model.included_complement
            frl.pos_list, frl.neg_list = frl._get_pos_neg_counts(X_, y)


def soft_falling_rule_list_rset(X,
                                y,
                                antecedents,
                                antecedent_map,
                                w,
                                C,
                                C1,
                                L_best,
                                epsilon,
                                curiosity_func,
                                max_iters=DEFAULT_MAX_ITERS,
                                terminate_prob=DEFAULT_TERMINATION_PROBABILITY,
                                verbose=False):
    """Find a sampling of the Rashomon set of soft falling rule lists.

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
    C1 : float
        The penalty for violating monotonicity in empirical positive proportion.
    L_best : float
        The best objective value found so far.
    epsilon : float
        The distance from the best objective value to be considered in the Rashomon set.
    curiosity_func : str
        The curiosity function to use.
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
    rule_lists = []

    # initialize a set to track unique models and a list for counts
    unique_models = set()
    unique_model_counts = []

    # for efficient lookup of rset statistics for curiosity functions
    trie = FallingRuleListTrie()

    # bitmasks for the indices of the positive and negative instances
    base_pos = index_list_to_bitmask(np.where(y == 1)[0], n)
    base_neg = index_list_to_bitmask(np.where(y == 0)[0], n)

    for i in tqdm(range(max_iters), "Optimizing softly falling rule list", disable=(not verbose)):
        prefix = []
        prefix_obj = 0
        # keep track of smallest empirical positive proportion so far
        alpha_min = 1

        # bitmasks for the remaining positive and negative instances after the current prefix
        pos_after_prefix = base_pos
        neg_after_prefix = base_neg
        n_pos_after_prefix = pos_after_prefix.bit_count()
        n_neg_after_prefix = neg_after_prefix.bit_count()

        # improvement is always possible at the beginning
        # so we will use a while True statement with a break later
        # to simulate a do-while loop
        # we only need to check the improvement condition after initializing a prefix
        while True:

            if np.random.rand() < terminate_prob:
                break

            candidate_antecedents = []
            curiosities = []
            terminate_if_selected = []

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

                # compute the objective function of the new rule
                loss_of_new_rule = _softFRL_loss_due_to_rule(n_pos_captured, n_neg_captured, alpha_min, w, C1, n)

                # get an objective lower bound from adding the rule
                bound_if_add_rule = softfrl_objective_lower_bound_addition(n_pos_after_new_rule, n_neg_after_new_rule,
                                                                           antecedent_alpha, w, C, C1, n)

                # get the objective if we add the rule and immediately terminate
                # this interacts with the termination condition in the while: True loop
                bound_if_terminate = _softFRL_loss_due_to_rule(n_pos_after_new_rule, n_neg_after_new_rule, alpha_min, w,
                                                               C1, n)
                bound = min(bound_if_add_rule, bound_if_terminate)

                if prefix_obj + loss_of_new_rule + bound < L_best + epsilon:
                    candidate_antecedents.append(antecedent)
                    tnode = trie.find_node(prefix, antecedent)
                    visits = tnode.visits if tnode else 0
                    total_count = trie.root.subtree_count
                    height = tnode.max_height if tnode else 0
                    if curiosity_func == 'paper':
                        curiosities.append(paper_curiosity_softFRL(alpha_min, n_pos_captured, n_pos_after_new_rule))
                    elif curiosity_func == "simulated_annealing":
                        curiosities.append(simulated_annealing_curiosity(alpha_min, i))
                    elif curiosity_func == "ucb":
                        curiosities.append(ucb_curiosity(alpha_min, visits, i))
                    elif curiosity_func == "single":
                        if len(antecedent) == 1:
                            curiosities.append(
                                paper_single_curiosity(alpha_min, n_pos_captured, n_pos_after_new_rule, 10,
                                                       single=True))
                        else:
                            curiosities.append(
                                paper_single_curiosity(alpha_min, n_pos_captured, n_pos_after_new_rule, 10))
                    elif curiosity_func == "depth":
                        curiosities.append(alpha_min + 1.0 / (height + 1))
                    elif curiosity_func == "hybrid":
                        if i < max_iters / 3:
                            curiosities.append(ucb_curiosity(alpha_min, visits, total_count))
                        else:
                            if height == 0:
                                curiosities.append(alpha_min + len(prefix))
                            else:
                                curiosities.append(alpha_min + len(prefix) + 1.0 / height)
                    terminate_if_selected.append(bound_if_terminate <= bound_if_add_rule)

            if not candidate_antecedents:
                break

            # choose the next antecedent and update the prefix and counts
            next_antecedent = _choose_next_antecedent(candidate_antecedents, curiosities)
            idx = candidate_antecedents.index(next_antecedent)
            terminate = terminate_if_selected[idx]

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
            alpha_min = min(alpha, alpha_min)

            prefix_obj += _softFRL_loss_due_to_rule(n_pos_captured, n_neg_captured, alpha_min, w, C1, n) + C
            prefix.append((next_antecedent, alpha))

            if terminate:
                break

        # calculate the objective including the remaining instances after the prefix
        loss_remaining = _softFRL_loss_due_to_rule(n_pos_after_prefix, n_neg_after_prefix, alpha_min, w, C1, n)
        obj = prefix_obj + loss_remaining

        if obj < L_best + epsilon:
            rule_list = prefix

            # append with empty tuple to indicate default "else clause" rule
            rule_list.append(((), n_pos_after_prefix / (n_pos_after_prefix + n_neg_after_prefix)))

            # insert the new rule list into the prefix trie
            trie.insert(rule_list, obj)
            # add the rule list to the Rashomon set
            rule_lists.append(rule_list)
            # update the unique models set
            unique_models.add(tuple(rule_list))

        unique_model_counts.append(len(unique_models))

    return rule_lists, unique_model_counts, unique_models


if __name__ == '__main__':
    import pandas as pd

    df = pd.read_csv('data/Broward Data.csv')
    X = df.iloc[:, :-1].astype(bool)
    y = df.iloc[:, -1]

    rset = SoftFRLRashomonSet()
    rset.fit(X, y, curiosity_func="ucb", verbose=True)
    print(len(rset.rset))
    print(len(set(rset.rset)))

    ref_obj = rset.reference_model.objective(X, y)
    rset_objs = [frl.objective(X, y) for frl in rset.rset]
    print(ref_obj)
    print(min(rset_objs))
    print(max(rset_objs))
