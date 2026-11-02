"""A module for computing bounds and checking feasibility.

These functions share common parameters.

Attributes:
    n (int): The total number of points in the dataset.
    n_pos_after (int): The number of points with positive labels after the rule list.
    n_neg_after (int): The number of points with negative labels after the rule list.
    last_alpha (float): The alpha value of the last antecedent in the rule list.
    w (float): The positive class weight.
    C (float): The penalty for adding a new rule.
"""


def objective_lower_bound_addition(n: int, n_pos_after: int, n_neg_after: int, last_alpha: float, w: float,
                                   C: float) -> float:
    """Compute the additive term of the lower bound from Theorem 4.6.
    """
    term1, term2, term3 = _compute_terms(n, n_pos_after, n_neg_after, last_alpha, w)
    return min(term3 + C, term1, term2)


def antecedent_is_feasible(n_neg_after: int, n_pos_after: int, antecedent_alpha: float, last_alpha: float,
                           w: float) -> bool:
    """Check if the antecedent is feasible according to Program 2.9 using Proposition 4.2.
    """
    alpha_after = (n_pos_after) / (n_pos_after + n_neg_after)
    return (antecedent_alpha <= last_alpha) \
        and (antecedent_alpha > (1 / (1 + w))) \
        and (alpha_after <= last_alpha)


def improvement_is_possible(n: int, n_pos_after: int, n_neg_after: int, last_alpha: float, w: float, C: float) -> bool:
    term1, term2, term3 = _compute_terms(n, n_pos_after, n_neg_after, last_alpha, w)
    """Check if improvement is possible with Inequality (9) from Theorem 4.6.
    """
    return (C < min(term1, term2) - term3)


def _compute_terms(n: int, n_pos_after: int, n_neg_after: int, last_alpha: float,
                   w: float) -> tuple[float, float, float]:
    """Compute the terms of the lower bound from Theorem 4.6.
    """
    if n == 0:
        return 0, 0, 0

    return (w / n) * n_pos_after, (1 / n) * n_neg_after, (1 / n) * ((1 / last_alpha) - 1) * n_pos_after
