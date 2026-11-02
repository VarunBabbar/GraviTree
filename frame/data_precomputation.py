"""Functions for precomputing data for the FRL algorithm.

These work by mining rules using the FPGrowth algorithm,
then precomputing a map from rules to all the points that satisfy them.
This is done to save computation time during the FRL algorithm,
at the cost of some storage.
This shouldn't be too bad if the number of rules and the number of points
are both reasonable.
"""
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth
from gmpy2 import mpz

MIN_SUPPORT = 0.02
MAX_LEN = 2


def mine_antecedents(X: pd.DataFrame | np.ndarray, min_support=MIN_SUPPORT, max_len=MAX_LEN) -> list[tuple]:
    """Mine frequent itemsets from the dataset.

    Parameters
    ----------
    X : pd.DataFrame | np.ndarray
        The dataset.
    min_support : float
        The minimum support threshold.

    Returns
    -------
    list
        The frequent itemsets.
    """
    if not isinstance(X, pd.DataFrame):
        X_ = pd.DataFrame(X)
    else:
        X_ = X

    frequent_itemsets = fpgrowth(X_, min_support=min_support, max_len=max_len, use_colnames=False)
    return [tuple(map(int, itemset)) for itemset in frequent_itemsets['itemsets']]


def get_points_satisfying_antecedent(X: pd.DataFrame | np.ndarray, antecedent: tuple) -> np.ndarray:
    """Get the indices of the points that satisfy the antecedent.

    Parameters
    ----------
    X : pd.DataFrame | np.ndarray
        The dataset.
    antecedent : tuple
        A tuple of indices of the columns that must be 1.

    Returns
    -------
    np.ndarray
        The indices of the points that satisfy the antecedent.
    """
    if isinstance(X, pd.DataFrame):
        X_ = X.values
    else:
        X_ = X

    return np.all(X_[:, antecedent] == 1, axis=1)


def build_antecedent_map(X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray, antecedents: list[tuple]) -> dict:
    """Build a map from antecedents to the indices of the points that satisfy them.

    Parameters
    ----------
    X : pd.DataFrame | np.ndarray
        The dataset.
    y : pd.Series | np.ndarray
        The labels.
    antecedents : list[tuple]
        A list of rules.

    Returns
    -------
    dict
        A map from rules to the indices of the points that satisfy them.
    """
    n = X.shape[0]
    antecedent_map = {}
    for antecedent in antecedents:
        points_satisfying_antecedent = get_points_satisfying_antecedent(X, antecedent)
        pos_bitmask = index_list_to_bitmask(np.where(points_satisfying_antecedent & (y == 1))[0], n)
        neg_bitmask = index_list_to_bitmask(np.where(points_satisfying_antecedent & (y == 0))[0], n)
        antecedent_map[antecedent] = {'pos': pos_bitmask, 'neg': neg_bitmask}
    return antecedent_map


def index_list_to_bitmask(index_list: list[int], n: int) -> int:
    """Convert a list of indices to a bitmask.

    Parameters
    ----------
    index_list : list
        The list of indices.
    n : int
        The length of the bitmask.

    Returns
    -------
    int
        The bitmask.
    """
    bitmask = 0
    for index in index_list:
        bitmask |= 1 << (n - int(index) - 1)
    return mpz(bin(bitmask))


if __name__ == '__main__':
    import pandas as pd

    df = pd.read_csv('data/compas.csv')
    X = df.iloc[:, :-1].astype(bool)
    y = df.iloc[:, -1]

    antecedents = mine_antecedents(X)
    antecedent_map = build_antecedent_map(X.values, y, antecedents)

    print(antecedents)
    print(antecedent_map)
