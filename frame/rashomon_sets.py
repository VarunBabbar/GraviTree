import numpy as np
from tqdm import tqdm
import time

from frame.optimization import DEFAULT_MAX_ITERS, DEFAULT_TERMINATION_PROBABILITY
from frame.optimization import captured_instances, remaining_instances, _choose_next_antecedent, _paper_curiosity, _loss_due_to_rule
from frame.bounds import objective_lower_bound_addition, antecedent_is_feasible, improvement_is_possible
from frame.data_precomputation import index_list_to_bitmask
from frame.FRL import FallingRuleList, DEFAULT_POSITIVE_WEIGHT, DEFAULT_RULE_PENALTY
from frame.Trie import TrieNode, FallingRuleListTrie
from frame.curiosity_functions import *
from collections import defaultdict

DEFAULT_EPSILON = 1e-2


class FRLRashomonSet:

    def __init__(self, w=DEFAULT_POSITIVE_WEIGHT, C=DEFAULT_RULE_PENALTY, epsilon=DEFAULT_EPSILON):
        self.w = w
        self.C = C
        self.epsilon = epsilon
        self.unique_model_counts = []  # list to track size of rset over iterations

    def fit(self,
            X,
            y,
            first_pass_iters=DEFAULT_MAX_ITERS,
            second_pass_iters=DEFAULT_MAX_ITERS,
            curiosity_func='paper',
            exploration_weight=1,
            exploitation_weight=1,
            gamma=0.8,
            best_obj=None,
            absolute_budget=None,
            min_support=None,
            max_len=None,
            **kwargs):
        self.reference_model = FallingRuleList(w=self.w, C=self.C)
        self.reference_model.fit(X,
                                 y,
                                 save_precomputation=True,
                                 max_iters=first_pass_iters,
                                 min_support=min_support,
                                 max_len=max_len,
                                 **kwargs)
        if best_obj == None:
            best_obj = self.reference_model.objective(X, y)
        epsilon_budget = self.epsilon
        if absolute_budget is not None:
            epsilon_budget = max(0.0, absolute_budget - best_obj)

        start_time = time.time()  # timing for second pass of algorithm
        rule_lists, unique_counts, betas_level = falling_rule_list_rset(X,
                                                                        y,
                                                                        self.reference_model.antecedents,
                                                                        self.reference_model.antecedent_map,
                                                                        self.w,
                                                                        self.C,
                                                                        best_obj,
                                                                        epsilon_budget,
                                                                        curiosity_func=curiosity_func,
                                                                        min_support=min_support,
                                                                        max_iters=second_pass_iters,
                                                                        exploitation_weight=exploitation_weight,
                                                                        exploration_weight=exploration_weight,
                                                                        gamma=gamma,
                                                                        **kwargs)
        end_time = time.time()
        self.betas_level = betas_level
        self.second_pass_time = end_time - start_time
        self.rset = [FallingRuleList.from_rule_list(rule_list, w=self.w, C=self.C) for rule_list in rule_lists]
        self.unique_model_counts = unique_counts
        self.unique_models = unique_counts[-1]

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


def falling_rule_list_rset(X,
                           y,
                           antecedents,
                           antecedent_map,
                           w,
                           C,
                           L_best,
                           epsilon,
                           curiosity_func,
                           min_support=None,
                           max_iters=DEFAULT_MAX_ITERS,
                           terminate_prob=DEFAULT_TERMINATION_PROBABILITY,
                           exploitation_weight=1,
                           exploration_weight=1,
                           gamma=1.0,
                           verbose=False):
    """Find a sampling of the Rashomon set of falling rule lists.

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
    L_best : float
        The target objective value.
    epsilon : float
        The tolerance for the objective value.
    max_iters : int, optional
        The maximum number of iterations.
    terminate_prob : float, optional
        The probability of terminating the optimization.
    verbose : bool, optional
        Whether to display a progress bar.

    Returns
    -------
    list[list[tuple]]
        A sampling of the Rashomon set of falling rule lists.
    """
    n = X.shape[0]
    rule_lists = []

    # initialize a set to track unique models and a list for counts
    unique_models = set()
    unique_model_counts = []

    state_counts = defaultdict(int)

    # for efficient lookup of rset statistics for curiosity functions
    trie = FallingRuleListTrie()

    # bitmasks for the indices of the positive and negative instances
    base_pos = index_list_to_bitmask(np.where(y == 1)[0], n)
    base_neg = index_list_to_bitmask(np.where(y == 0)[0], n)
    betas_level = defaultdict(list)

    for i in tqdm(range(max_iters), "Optimizing rule list", disable=(not verbose)):
        prefix = []
        prefix_without_probabilities = []
        prefix_obj = 0
        alpha = 1

        # bitmasks for the remaining positive and negative instances after the current prefix
        pos_after_prefix = base_pos
        neg_after_prefix = base_neg
        counter = 0
        n_pos_after_prefix = pos_after_prefix.bit_count()
        n_neg_after_prefix = neg_after_prefix.bit_count()

        while improvement_is_possible(n, n_pos_after_prefix, n_neg_after_prefix, alpha, w, C):
            if np.random.rand() < terminate_prob:
                break

            candidate_antecedents = []
            curiosities = []
            beta = 0

            for antecedent in antecedents:
                pos_captured_full_data, neg_captured_full_data = antecedent_map[antecedent]['pos'], antecedent_map[
                    antecedent]['neg']

                # find the instances captured by the antecedent
                pos_captured, neg_captured, n_pos_captured, n_neg_captured = captured_instances(
                    pos_captured_full_data, neg_captured_full_data, pos_after_prefix, neg_after_prefix)
                n_captured = n_pos_captured + n_neg_captured

                # find the instances left uncaptured by the antecedent and the rest of the prefix
                _, _, n_pos_after_new_rule, n_neg_after_new_rule = remaining_instances(
                    pos_captured, neg_captured, pos_after_prefix, neg_after_prefix)
                n_after_new_rule = n_pos_after_new_rule + n_neg_after_new_rule

                if n_captured == 0 or n_after_new_rule == 0:
                    continue
                if min_support is not None:
                    min_count = int(np.ceil(min_support * n))
                    if n_captured < min_count or n_after_new_rule < min_count:
                        continue

                antecedent_alpha = n_pos_captured / n_captured

                if antecedent_is_feasible(n_neg_after_new_rule, n_pos_after_new_rule, antecedent_alpha, alpha, w):
                    loss_of_new_rule = _loss_due_to_rule(n_pos_captured, n_neg_captured, w, n)
                    bound = objective_lower_bound_addition(n, n_pos_after_new_rule, n_neg_after_new_rule, alpha, w, C)
                    # adjustment 1 from the original algorithm
                    # compares to the target objective value L_best instead of the best found so far
                    if prefix_obj + loss_of_new_rule + C + bound < L_best + epsilon:
                        candidate_antecedents.append(antecedent)
                        state = tuple(prefix_without_probabilities)
                        action = antecedent
                        tnode = trie.find_node(prefix, antecedent)
                        beta += 1
                        visits = tnode.visits if tnode else 0
                        rset_size = trie.root.subtree_count
                        subtree_count = tnode.subtree_count if tnode else 0
                        height = tnode.max_height if tnode else 0
                        if curiosity_func == 'paper':
                            curiosities.append(
                                _paper_curiosity(antecedent_alpha, n_pos_captured, n_pos_after_new_rule, gamma))
                        elif curiosity_func == "simulated_annealing":
                            curiosities.append(simulated_annealing_curiosity(antecedent_alpha, i))
                        elif curiosity_func == "ucb":
                            curiosities.append(
                                ucb_curiosity(antecedent_alpha, visits, i, exploration_weight=exploration_weight))
                        elif curiosity_func == 'ucb_marginal':
                            curiosities.append(
                                ucb_curiosity_marginal(antecedent_alpha,
                                                       state_counts[(state, action)],
                                                       i,
                                                       exploration_weight=exploration_weight))
                        elif curiosity_func == "ucb+":
                            curiosities.append(
                                ucb_reward(antecedent_alpha,
                                           visits,
                                           rset_size,
                                           subtree_count,
                                           i,
                                           exploration_weight=exploration_weight,
                                           exploitation_weight=exploitation_weight))
                        elif curiosity_func == "single":
                            if len(antecedent) == 1:
                                curiosities.append(
                                    paper_single_curiosity(antecedent_alpha,
                                                           n_pos_captured,
                                                           n_pos_after_new_rule,
                                                           10,
                                                           single=True))
                            else:
                                curiosities.append(
                                    paper_single_curiosity(antecedent_alpha, n_pos_captured, n_pos_after_new_rule, 10))
                        elif curiosity_func == "depth":
                            curiosities.append(antecedent_alpha + 1.0 / (height + 1))
                        elif curiosity_func == "hybrid":
                            if i < max_iters / 3:
                                curiosities.append(ucb_curiosity(antecedent_alpha, visits, rset_size))
                            else:
                                if height == 0:
                                    curiosities.append(antecedent_alpha + len(prefix))
                                else:
                                    curiosities.append(antecedent_alpha + len(prefix) + 1.0 / height)
                        elif curiosity_func == 'uniform':
                            curiosities.append(1.0)

            if not candidate_antecedents:
                break

            # choose the next antecedent and update the prefix and counts
            next_antecedent = _choose_next_antecedent(candidate_antecedents, curiosities)

            pos_captured_full_data, neg_captured_full_data = antecedent_map[next_antecedent]['pos'], antecedent_map[
                next_antecedent]['neg']

            pos_captured, neg_captured, n_pos_captured, n_neg_captured = captured_instances(
                pos_captured_full_data, neg_captured_full_data, pos_after_prefix, neg_after_prefix)

            pos_after_prefix, neg_after_prefix, n_pos_after_prefix, n_neg_after_prefix = remaining_instances(
                pos_captured, neg_captured, pos_after_prefix, neg_after_prefix)

            n_captured = n_pos_captured + n_neg_captured
            alpha = n_pos_captured / n_captured

            prefix_obj += _loss_due_to_rule(n_pos_captured, n_neg_captured, w, n) + C
            prefix.append((next_antecedent, alpha))
            state = tuple(prefix_without_probabilities)
            action = next_antecedent
            state_counts[(state, action)] += 1
            prefix_without_probabilities.append(next_antecedent)
            beta /= len(antecedents)
            betas_level[counter].append(beta)
            counter += 1

        # calculate the objective including the remaining instances after the prefix
        loss_remaining = _loss_due_to_rule(n_pos_after_prefix, n_neg_after_prefix, w, n)
        obj = prefix_obj + loss_remaining

        # adjustment 2 from the original algorithm
        # compares to the target objective value L_best instead of the best found so far
        if obj < L_best + epsilon:
            rule_list = prefix
            rule_list.append(((), n_pos_after_prefix / (n_pos_after_prefix + n_neg_after_prefix)))
            if tuple(rule_list) not in unique_models:
                if curiosity_func != "paper":
                    trie.insert(rule_list, obj)  # update visits
                rule_lists.append(rule_list)
                unique_models.add(tuple(rule_list))

        unique_model_counts.append(len(unique_models))

    return rule_lists, unique_model_counts, betas_level


def sorted_rule_tuple(rule_list):
    rules = [x[0] for x in rule_list]
    tup = tuple(sorted(rules))
    return tup


if __name__ == '__main__':
    import pandas as pd

    df = pd.read_csv('data/Australian Credit.csv')
    X = df.iloc[:, :-1].astype(bool)
    y = df.iloc[:, -1]

    rset = FRLRashomonSet()
    rset.fit(X, y, curiosity_func="reward", verbose=True)
    print(len(rset.rset))
    print(rset.rset[0].rule_list)

    ref_obj = rset.reference_model.objective(X, y)
    rset_objs = [frl.objective(X, y) for frl in rset.rset]
    print(ref_obj)
    print(min(rset_objs))
    print(max(rset_objs))
