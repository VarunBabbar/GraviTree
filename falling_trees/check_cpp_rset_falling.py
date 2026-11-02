import json
import pickle
import sys
from typing import Any, Dict

# Ensure falling-models is importable when this module is imported standalone
PARENT_DIR = "/usr/xtmp/vb97/FRL_Rashomon_Set/falling-models"
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from falling_trees.frl_rashomon_set_alg_falling_constraint_v2 import (
    Leaf,
    Node,
    is_falling_tree,
    is_falling_tree_new_constraint,
)


def tree_key(tree):
    if tree is None:
        return ("NIL",)
    if isinstance(tree, Leaf):
        return ("L", float(tree.pred_prob), float(tree.objective))
    return ("N", int(tree.feature), float(tree.objective), tree_key(tree.left), tree_key(tree.right))


def tree_from_json(obj: Dict[str, Any]):
    if obj is None:
        return None
    t = obj.get("t")
    if t == "L":
        return Leaf(pred_prob=float(obj["p"]), objective=float(obj["o"]))
    if t == "N":
        left = tree_from_json(obj["l"])
        right = tree_from_json(obj["r"])
        return Node(feature=int(obj["f"]), left=left, right=right, objective=float(obj["o"]))
    raise ValueError(f"Unknown node type: {t}")


def load_cpp_rset(jsonl_path: str):
    results = []
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            tree = tree_from_json(entry["tree"])
            results.append((tree, float(entry["obj"]), float(entry["pmax"])))
    return results


def main():
    jsonl_path = (
        "/usr/xtmp/vb97/FRL_Rashomon_Set/falling-models/falling_trees/cpp_rset_trees.jsonl"
    )
    pkl_path = (
        "/usr/xtmp/vb97/FRL_Rashomon_Set/falling-models/falling_trees/cpp_rset_trees.pkl"
    )
    rset = load_cpp_rset(jsonl_path)
    # with open(pkl_path, "wb") as f:
    #     pickle.dump(rset, f)
    unique_keys = {tree_key(t) for (t, _, _) in rset}
    dup_count = len(rset) - len(unique_keys)
    non_falling = [t for (t, _, _) in rset if not is_falling_tree(t)]
    non_falling_new = [t for (t, _, _) in rset if not is_falling_tree_new_constraint(t)]
    print(f"Loaded {len(rset)} trees from {jsonl_path}")
    print(f"Repeated trees (structural): {dup_count}")
    print(f"Saved pickle to {pkl_path}")
    print(f"Non-falling (old constraint): {len(non_falling)}")
    print(f"Non-falling (new constraint): {len(non_falling_new)}")


if __name__ == "__main__":
    main()

