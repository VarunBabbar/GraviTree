import numpy as np
import math

DEFAULT_LAMBDA = 0.8

def simulated_annealing_curiosity(alpha_rule, iteration, initial_temp=1.0, cooling_rate=0.99):
    """
    Compute the curiosity score for a rule using a simulated annealing approach.

    Parameters:
        alpha_rule : float
            Quality measure of the rule (e.g., empirical positive proportion).
        iteration : int
            The current iteration number, used to compute the current temperature.
        initial_temp : float, optional
            The starting temperature for annealing.
        cooling_rate : float, optional
            The rate at which the temperature decreases.

    Returns:
        float: A curiosity score that balances quality with exploration.
    """
    temperature = initial_temp * (cooling_rate ** iteration)
    quality = alpha_rule
    random_component = np.random.rand() * temperature
    return quality + random_component

import numpy as np

def ucb_curiosity(alpha_rule, count, total_count, exploration_weight=1.0):
    """
    Returns a curiosity score using an Upper Confidence Bound (UCB) style metric.
    
    Parameters:
        alpha_rule (float): Quality measure of the candidate rule.
        n_pos_captured (int): Number of positive instances captured.
        n_pos_after (int): Number of positive instances remaining after the rule.
        count (int): Number of times this candidate has been chosen.
        total_count (int): Total number of candidate evaluations so far.
        exploration_weight (float): Weight for the exploration term.
    
    Returns:
        float: Curiosity score.
    """
    if count == 0:
        # Encourage exploration for never-chosen candidates
        return exploration_weight
    # UCB-like term: increase curiosity for candidates that have been chosen less frequently
    bonus = exploration_weight * np.sqrt(np.log(total_count) / count)
    return alpha_rule + bonus

def ucb_curiosity_marginal(alpha_rule,count,total_count, exploration_weight=1.0):
    # count and total_count are defined differently here
    return ucb_curiosity(alpha_rule, count, total_count, exploration_weight)

def ucb_reward(alpha_rule, count, total_count, rset_size, subtree_count, exploration_weight=1.0, exploitation_weight=1.0):
    """
    Returns a curiosity score using an Upper Confidence Bound (UCB) style metric, plus an extra reward.
    
    Parameters:
        alpha_rule (float): Quality measure of the candidate rule.
        n_pos_captured (int): Number of positive instances captured.
        n_pos_after (int): Number of positive instances remaining after the rule.
        count (int): Number of times this candidate has been chosen.
        total_count (int): Total number of candidate evaluations so far.
        rset_size (int): Size of the Rashomon Set at the current iteration.
        subtree_count (int): Number of models in the Rashomon Set stemming from the prefix, or number of leaves in the subtree. 
        exploration_weight (float): Weight for the exploration term.
        exploitation_weight (float): Weight for exploitation term.
    
    Returns:
        float: Curiosity score.
    """
    if count == 0 or rset_size == 0:
        # Encourage exploration for never-chosen candidates
        return exploration_weight
    # UCB-like term: increase curiosity for candidates that have been chosen less frequently
    bonus = exploration_weight * np.sqrt(np.log(total_count) / count)
    reward = exploitation_weight * np.sqrt(subtree_count / rset_size)
    return alpha_rule + bonus + reward

def paper_single_curiosity(alpha_rule: float, n_pos_captured_by_rule: int, n_pos_after: int, single_weight, single=False, lmbda=DEFAULT_LAMBDA):
    """Compute the curiosity of a rule, giving a boost to rules with only one condition.
    
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
    return curiosity * single_weight if single else curiosity

def depth_curiosity(alpha_rule: float, n_pos_captured_by_rule: int, n_pos_after: int, num_models_with_prefix: int, trie_height, lmbda=DEFAULT_LAMBDA):
    """Compute the curiosity of a rule, discouraging previously prefixes in the rset.
    
    Parameters
    ----------
    alpha_rule : float
        The empirical positive proportion of the rule.
    n_pos_captured_by_rule : int
        The number of positive instances captured by the rule.
    n_pos_after : int
        The number of positive instances after the rule.
    num_models_with_prefix : int
        The number of models in the rset starting with a prefix
    lmbda : float, optional
        The tradeoff between emp pos prop and ratio of remaining pos instances captured by the rule.

    Returns
    -------
    float
        The curiosity of the rule.
    """
    curiosity = (lmbda * alpha_rule) + ((1 - lmbda) * (n_pos_captured_by_rule / n_pos_after))
    return curiosity * (math.factorial(trie_height) / (1 + num_models_with_prefix))