import pandas as pd
import numpy as np
import pickle


# Load the helper files
with open('experiment_15_helpers/compas_cols.pkl', 'rb') as file:
    compas_cols = pickle.load(file)
with open('experiment_15_helpers/ds_features.pkl', 'rb') as file:
    ds_features = pickle.load(file)
with open('experiment_15_helpers/feature_proba.pkl', 'rb') as file:
    feature_proba = pickle.load(file)
with open('experiment_15_helpers/logical_map.pkl', 'rb') as file:
    logical_map = pickle.load(file)



def compas_bootstrap(X, dataset, feat):
    '''
    Generate copy of X with bootstrapped feature (as if non-binarized)
    Assume compas_cols, ds_features and feature_proba already exist globally

    params
    --------
    X: pd.DataFrame
        the boolean design matrix *with* negations already present
    dataset: str
        dataset file name (for looking up ds_features, maps, etc)
    feat: str
        non-binarized feature name
    '''
    X_boot = X.copy(deep=True)
    n = len(X_boot)

    if feat == 'Race':
        race_cols = list(feature_proba[dataset]['Race'].keys())
        probs = np.array([feature_proba[dataset]['Race'][r] for r in race_cols], dtype=float)
        probs /= probs.sum()
        picks = np.random.choice(race_cols, size=n, p=probs)
        for col in race_cols:
            mask = (picks == col)
            X_boot[col] = mask
            X_boot['~' + col] = ~mask  # update its negation
    elif feat == 'Gender=Male':
        col_idx = X.columns.get_loc(feat)
        col_idx_neg = X.columns.get_loc('~'+feat)
        sample_idx = np.random.choice(len(X), len(X), replace=True)
        X_boot.iloc[:, col_idx] = X.iloc[sample_idx, col_idx].to_numpy()
        X_boot.iloc[:, col_idx_neg] = X.iloc[sample_idx, col_idx_neg].to_numpy()
    elif feat == 'Age':
        arr = compas_cols['age'].to_numpy()
        samp = np.random.choice(arr, size=n, replace=True)

        # assign each Age threshold
        m1 = (samp >= 18) & (samp <= 20)
        m2 = (samp >= 18) & (samp <= 22)
        m3 = (samp >= 18) & (samp <= 25)
        m4 = (samp >= 24) & (samp <= 30)
        m5 = (samp >= 24) & (samp <= 40)
        m6 = samp >= 30
        m7 = samp >= 40
        m8 = samp >= 45

        X_boot['Age=18-20'] = m1;    X_boot['~Age=18-20'] = ~m1
        X_boot['Age=18-22'] = m2;    X_boot['~Age=18-22'] = ~m2
        X_boot['Age=18-25'] = m3;    X_boot['~Age=18-25'] = ~m3
        X_boot['Age=24-30'] = m4;    X_boot['~Age=24-30'] = ~m4
        X_boot['Age=24-40'] = m5;    X_boot['~Age=24-40'] = ~m5
        X_boot['Age>=30']   = m6;    X_boot['~Age>=30']   = ~m6
        X_boot['Age>=40']   = m7;    X_boot['~Age>=40']   = ~m7
        X_boot['Age>=45']   = m8;    X_boot['~Age>=45']   = ~m8

    elif feat in ['Current-Charge-Degree=Misdemeanor', 'Juvenile-Felonies']:
        arr = compas_cols['juv_fel_count'].to_numpy()
        samp = np.random.choice(arr, size=n, replace=True)

        m0 = samp == 0
        m1 = (samp >= 1) & (samp <= 3)
        m2 = samp > 3

        X_boot['Juvenile-Felonies=0']   = m0; X_boot['~Juvenile-Felonies=0']   = ~m0
        X_boot['Juvenile-Felonies=1-3'] = m1; X_boot['~Juvenile-Felonies=1-3'] = ~m1
        X_boot['Juvenile-Felonies>3']   = m2; X_boot['~Juvenile-Felonies>3']   = ~m2

    elif feat == 'Juvenile-Crimes':
        arr = compas_cols['juv_crimes'].to_numpy()
        samp = np.random.choice(arr, size=n, replace=True)

        m0 = samp == 0
        m1 = (samp >= 1) & (samp <= 3)
        m2 = samp > 3
        m3 = samp > 5

        X_boot['Juvenile-Crimes=0']   = m0; X_boot['~Juvenile-Crimes=0']   = ~m0
        X_boot['Juvenile-Crimes=1-3'] = m1; X_boot['~Juvenile-Crimes=1-3'] = ~m1
        X_boot['Juvenile-Crimes>3']   = m2; X_boot['~Juvenile-Crimes>3']   = ~m2
        X_boot['Juvenile-Crimes>5']   = m3; X_boot['~Juvenile-Crimes>5']   = ~m3

    elif feat == 'Prior-Crimes':
        arr = compas_cols['priors_count'].to_numpy()
        samp = np.random.choice(arr, size=n, replace=True)

        m0 = samp == 0
        m1 = (samp >= 1) & (samp <= 3)
        m2 = samp > 3
        m3 = samp > 5

        X_boot['Prior-Crimes=0']   = m0; X_boot['~Prior-Crimes=0']   = ~m0
        X_boot['Prior-Crimes=1-3'] = m1; X_boot['~Prior-Crimes=1-3'] = ~m1
        X_boot['Prior-Crimes>3']   = m2; X_boot['~Prior-Crimes>3']   = ~m2
        X_boot['Prior-Crimes>5']   = m3; X_boot['~Prior-Crimes>5']   = ~m3

    else:
        raise ValueError(f"Unhandled feat {feat} in compas_bootstrap")

    return X_boot

def gen_bootstrap(X, dataset, feat):
    '''
    Generate copy of X with bootstrapped feature (as if non-binarized)
    Assume logical_map, ds_features and feature_proba already exist globally

    params
    --------
    X: pd.DataFrame
        the boolean design matrix *with* negations already present
    dataset: str
        dataset file name (for looking up ds_features, maps, etc)
    feat: str
        non-binarized feature name
    '''
    single_feats = set(X.columns) & set(ds_features[dataset])
    group_feats = set(ds_features[dataset]) - set(X.columns)

    if feat in single_feats:
        sample_idx = np.random.choice(len(X), len(X), replace=True)
        X_boot = X.copy(deep=True)
        X_boot[feat] = X[feat].iloc[sample_idx].to_numpy()
        X_boot['~'+feat] = X['~'+feat].iloc[sample_idx].to_numpy()

    elif feat in group_feats:
        choice_map = logical_map[dataset][feat]

        choices = list(choice_map.keys())
        probs = np.array([feature_proba[dataset][feat][ch] for ch in choices], dtype=float)

        total = probs.sum()
        probs /= total

        X_boot = X.copy(deep=True)
        # for each row, sample a subgroup (or none) and assign all sub-feats
        for i in range(len(X_boot)):
            pick   = np.random.choice(choices, p=probs)
            assign = logical_map[dataset][feat][pick]
            for subfeat, val in assign.items():
                X_boot.iat[i, X_boot.columns.get_loc(subfeat)] = val

        # suppose `orig` is the list of the *base* columns you want to negate
        orig = [c for c in X_boot.columns if not c.startswith('~')]
        # build a DataFrame of all negations at once
        neg_df = (~X_boot[orig]).rename(
            columns={c: f"~{c}" for c in orig}
        )
        # concat in one go
        X_boot = pd.concat([X_boot, neg_df], axis=1)

    else:
        raise ValueError(f"Feature {feat} is not recognized in dataset {dataset}")

    return X_boot
