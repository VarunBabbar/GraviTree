import os
import sys
from sklearn.model_selection import train_test_split

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from falling_trees.gravitree import GraviTree
from falling_trees.binarize_dataset import binarize_dataset
from utils import predict_falling_tree


def binarize_data(X_train, y_train, X_test, y_test):
    """Binarize the dataset using the falling_trees binarization function."""
    train_df = pd.DataFrame(X_train, columns=[f"feature_{i}" for i in range(X_train.shape[1])])
    train_df['label'] = y_train
    test_df = pd.DataFrame(X_test, columns=[f"feature_{i}" for i in range(X_test.shape[1])])
    test_df['label'] = y_test

    train_df, thresholds, header, _ = binarize_dataset(train_df, num_estimators=40)
    test_df = binarize_dataset(test_df, num_estimators=100, thresholds=thresholds, header=header)

    X_train_df = train_df.drop(columns=['label'])
    X_train = X_train_df.values
    y_train = train_df['label'].astype(int).values
    X_test_df = test_df.drop(columns=['label'])
    X_test = X_test_df.values
    y_test = test_df['label'].astype(int).values

    return X_train, y_train, X_test, y_test, thresholds, header


if __name__ == "__main__":
    import pandas as pd
    # Load the dataset
    data = pd.read_csv("/home/users/zwb3/research/falling-models/data/benchmark/compas.csv")
    X = data.drop(columns=["two_year_recid"])
    original_features = X.columns.tolist()
    X = X.values
    y = data["two_year_recid"].values

    print(f"Running experiment with labels two_year_recid...")

    # make a test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train, y_train, X_test, y_test, thresholds, header = binarize_data(X_train, y_train, X_test, y_test)

    # Create and fit the GraviTree model
    model = GraviTree()
    model.fit(X_train, y_train)
    print(len(model.ft_rset), "trees in the R-set.")
    print(model.score(X_test, y_test))
