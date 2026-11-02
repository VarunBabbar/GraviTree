import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from frame.data_precomputation import mine_antecedents, build_antecedent_map, MIN_SUPPORT, MAX_LEN
from frame.optimization import optimize_falling_rule_list
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DEFAULT_POSITIVE_WEIGHT = 1
DEFAULT_RULE_PENALTY = 1e-6


class FallingRuleList(BaseEstimator, ClassifierMixin):

    def __init__(self, w=DEFAULT_POSITIVE_WEIGHT, C=DEFAULT_RULE_PENALTY):
        self.w = w
        self.C = C

    @classmethod
    def from_rule_list(cls, rule_list, w=DEFAULT_POSITIVE_WEIGHT, C=DEFAULT_RULE_PENALTY):
        frl = cls(w=w, C=C)
        frl.rule_list = rule_list
        return frl

    def fit(self,
            X,
            y,
            opt_curiosity="paper",
            verbose=False,
            include_complement=True,
            save_precomputation=False,
            min_support=None,
            max_len=None,
            **kwargs):
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
                X['~' + col] = 1 - X[col]

        self.features = X.columns
        if verbose:
            logging.info(f"Features: {self.features}")

        # mine frequent itemsets
        if verbose:
            logging.info("Mining antecedents")
        ms = MIN_SUPPORT if min_support is None else min_support
        ml = MAX_LEN if max_len is None else max_len
        antecedents = mine_antecedents(X, min_support=ms, max_len=ml)
        if verbose:
            logging.info(f"Number of mined antecedents (min_support={ms}, max_len={ml}): {len(antecedents)}")

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
            logging.info("Optimizing falling rule list")
        self.rule_list, best_obj = optimize_falling_rule_list(X,
                                                              y,
                                                              antecedents,
                                                              antecedent_map,
                                                              self.w,
                                                              self.C,
                                                              opt_curiosity=opt_curiosity,
                                                              verbose=verbose,
                                                              **kwargs)

        if verbose:
            logging.info("Falling rule list optimization complete")

        pos, neg = self._get_pos_neg_counts(X, y)
        self.pos_list = pos
        self.neg_list = neg
        self.best_obj = best_obj

        return self

    def predict(self, X, threshold=None):
        probs = self.predict_proba(X)
        # 1 / (1 + w) is proven to be the optimal threshold in Theorem 2.8
        # use it as the default threshold
        threshold = threshold or 1 / (1 + self.w)
        return (probs > threshold).astype(int)

    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            X_ = X.values
        else:
            X_ = X

        n = X_.shape[0]
        preds = np.zeros(n)
        already_captured = np.zeros(n, dtype=bool)

        for (rule, prob) in self.rule_list:
            captured = np.all(X_[:, rule] == 1, axis=1)
            # only update predictions for points that haven't been captured yet
            captured_by_rule = captured & ~already_captured
            preds[captured_by_rule] = prob
            already_captured |= captured

        return preds

    def score(self, X, y):
        return np.mean(self.predict(X) == y)

    def weighted_score(self, X, y):
        preds = self.predict(X)
        return np.mean((preds == y) * (1 + self.w * y))

    def objective(self, X, y):
        preds = self.predict(X)
        misclass = (preds != y).astype(float)
        weighted_misclass = (1 / X.shape[0]) * (np.sum(misclass * (y == 0)) + self.w * np.sum(misclass * (y == 1)))
        return weighted_misclass + self.C * (len(self.rule_list) - 1)

    def _rule_string(self, rule):
        return ' & '.join([self.features[i] for i in rule])

    def _make_rule_df(self):
        table_rep = pd.DataFrame(columns=['Rule', 'Probability'], data=self.rule_list)
        table_rep['Rule'] = table_rep['Rule'].apply(self._rule_string)
        table_rep['Probability'] = table_rep['Probability'].apply(lambda x: f'{x:.3f}')
        if len(table_rep) == 1:
            return table_rep

        table_rep[''] = ['if'] + ['else if'] * (len(table_rep) - 2) + ['else']
        table_rep = table_rep[['', 'Rule', 'Probability']]

        table_rep['+'] = self.pos_list
        table_rep['-'] = self.neg_list

        return table_rep

    def _get_pos_neg_counts(self, X, y):
        if isinstance(X, pd.DataFrame):
            X_ = X.values
        else:
            X_ = X

        pos_capture = []
        neg_capture = []
        already_captured = np.zeros(X_.shape[0], dtype=bool)
        for (rule, _) in self.rule_list:
            captured = np.all(X_[:, rule] == 1, axis=1) & ~already_captured
            pos_capture.append(np.sum(captured & (y == 1)))
            neg_capture.append(np.sum(captured & (y == 0)))
            already_captured |= captured

        return pos_capture, neg_capture

    def __eq__(self, other):
        return (self.rule_list == other.rule_list) and (self.features == other.features).all()

    def __hash__(self):
        return hash(str(self.rule_list) + str(self.features))

    def __str__(self):
        return self._make_rule_df().to_string(index=False)

    def __repr__(self):
        return f'w: {self.w}\nC: {self.C}\n\n' + str(self)


if __name__ == '__main__':
    import pandas as pd

    df = pd.read_csv('data/compas.csv')
    X = df.iloc[:, :-1].astype(bool)
    y = df.iloc[:, -1]

    frl = FallingRuleList()
    frl.fit(X, y, verbose=True)
    print(frl)

    print(frl.score(X, y))

    df = pd.read_csv('data/Bank Full Clean Uncut.csv')
    X = df.iloc[:, :-1].astype(bool)
    y = df.iloc[:, -1]

    frl = FallingRuleList(w=7, C=1e-7)
    frl.fit(X, y, verbose=True, include_complement=False, max_iters=3000)
    print(frl)

    print()

    print(frl.score(X, y))
    print(frl.weighted_score(X, y))
