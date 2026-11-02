import timbertrek
from treefarms import TREEFARMS
import numpy as np
from tqdm import tqdm
import time
from json import load

np.random.seed(3)
dataset = "NIJ Recidivism.csv" # can switch to any other dataset

df = pd.read_csv(f'data/{dataset}')
X = df.iloc[:, :-1].astype(bool)
y = df.iloc[:, -1]

subsampled_dfs = []
config = {
    "regularization": 0.005,
    "rashomon_bound_adder": 0.02,
    "depth_budget": 5,
    "verbose": False,
    "time_limit": 100,
    "rashomon_trie": "trie_{}.json".format(dataset.split('.')[0])
}

start_time = time.time()
tf = TREEFARMS(config)
tf.fit(X,y)
end_time = time.time()
rset_runtime = end_time-start_time

# left = true
# -2 = true
# -1 = false
def get_trie_of_frls(trie, df, y, parent_prob=1.1):
    frl_trie = {}
    for key in trie:
        keysplit = key.strip().split()
        
        # Determine which rows satisfy this rule list prefix
        if key == "-2 -1":  # Terminal node
            frl_trie[key] = trie[key]
            continue
        
        if len(keysplit) == 1:
            feature = int(keysplit[0])
            mask = df.iloc[:, feature] == 1
        else:
            if keysplit[0] != '-2':# can also have and != -1
                continue
            mask = (df.iloc[:, int(keysplit[1])] == 1)

        labels = y[mask]
        if len(labels) == 0:
            continue
        
        prob = labels.mean()

        # Enforce monotonicity: prob must be <= parent_prob
        if prob > parent_prob:
            continue
        
        # Recursively prune children
        pruned_subtrie = get_trie_of_frls(trie[key].copy(), df[~mask], y[~mask], parent_prob=prob)
        if len(pruned_subtrie) > 0:
            frl_trie[key] = pruned_subtrie
        else:
            # Still keep this node if it's a leaf
            if "-2 -1" in trie[key]: # or "-1 -2" in trie[key]
                frl_trie[key] = trie[key]
    
    return frl_trie

trie = load(open("trie_{}.json".format(dataset.split('.')[0]), "r"))

new_frl_trie = get_trie_of_frls(trie,df,y)

decision_paths = timbertrek.transform_trie_to_rules(
    new_frl_trie,
    df,
    feature_names=feature_names,
)
