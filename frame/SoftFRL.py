import logging
import pandas as pd
from frame.data_precomputation import mine_antecedents, build_antecedent_map
from frame.optimization import optimize_soft_falling_rule_list

from frame.FRL import FallingRuleList, DEFAULT_POSITIVE_WEIGHT, DEFAULT_RULE_PENALTY

DEFAULT_FALLING_PENALTY = 0.5


class SoftFRL(FallingRuleList):

    def __init__(self, w=DEFAULT_POSITIVE_WEIGHT, C=DEFAULT_RULE_PENALTY, C1=DEFAULT_FALLING_PENALTY):
        super().__init__(w=w, C=C)
        self.C1 = C1

    @classmethod
    def from_rule_list(cls, rule_list, w=DEFAULT_POSITIVE_WEIGHT, C=DEFAULT_RULE_PENALTY, C1=DEFAULT_FALLING_PENALTY):
        frl = cls(w=w, C=C, C1=C1)
        frl.rule_list = rule_list
        return frl

    def fit(self, X, y, verbose=False, include_complement=True, save_precomputation=False, **kwargs):
        self.included_complement = include_complement

        if verbose:
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
            logging.info("Starting fit method")

        if include_complement:
            if verbose:
                logging.info("Including complement of features")
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X)

            for col in X.columns:
                X['~' + col] = ~X[col]

        self.features = X.columns
        if verbose:
            logging.info(f"Features: {self.features}")

        # mine frequent itemsets
        if verbose:
            logging.info("Mining antecedents")
        antecedents = mine_antecedents(X)
        if verbose:
            logging.info(f"Number of mined antecedents: {len(antecedents)}")

        # precompute map from antecedents to captured instances
        # we don't keep this around after fitting because it's too large
        if verbose:
            logging.info("Building antecedent map")
        antecedent_map = build_antecedent_map(X, y, antecedents)

        if save_precomputation:
            self.antecedents = antecedents
            self.antecedent_map = antecedent_map

        # run rule list optimization algorithm
        if verbose:
            logging.info("Optimizing soft falling rule list")

        self.rule_list, best_obj = optimize_soft_falling_rule_list(X,
                                                                   y,
                                                                   antecedents,
                                                                   antecedent_map,
                                                                   self.w,
                                                                   self.C,
                                                                   self.C1,
                                                                   verbose=verbose,
                                                                   **kwargs)

        if verbose:
            logging.info("Soft falling rule list optimization complete")

        pos, neg = self._get_pos_neg_counts(X, y)
        self.pos_list = pos
        self.neg_list = neg
        self.best_obj = best_obj

        return self


if __name__ == "__main__":
    # Example usage
    df = pd.read_csv('data/compas.csv')
    X = df.iloc[:, :-1].astype(bool)
    y = df.iloc[:, -1]

    model = SoftFRL(C1=0.5)
    model.fit(X, y, verbose=True, max_iters=6000)
    print("Rule List:", model)
    print("Best Objective:", model.best_obj)
    print("Positive List:", model.pos_list)
