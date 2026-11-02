class TrieNode:

    def __init__(self, rule, probability=0.0, loss=None):
        """
        rule: int, a positive integer for a rule, or -1/-2 for leaves (else statements).
        pos_count: number of positive examples captured by this rule.
        neg_count: number of negative examples captured by this rule.
        probability: probability associated with this rule.
        loss: only defined for leaf nodes to store the loss for the entire rule list.
        """
        self.rule = rule
        self.probability = probability
        self.loss = loss
        # children maps a rule (int) to its corresponding TrieNode.
        self.children = {}
        # Additional attributes for the Rashomon set representation:
        self.subtree_count = 0  # Number of complete rule lists (leaves) in this subtree.
        self.max_height = 0  # Maximum height (number of rules from here to a leaf).
        self.visits = 0

    def add_child(self, child_node):
        """Adds a child node keyed by its rule."""
        self.children[child_node.rule] = child_node

    def __repr__(self):
        return (f"TrieNode(rule={self.rule},"
                f"prob={self.probability:.2f}, loss={self.loss}, count={self.subtree_count}, "
                f"max_height={self.max_height}, children={list(self.children.keys())})")


class FallingRuleListTrie:

    def __init__(self):
        # The root is a dummy node.
        self.root = TrieNode(rule=None)

    def insert(self, rule_list, loss):
        """
        Inserts a falling rule list into the trie while updating the count and height attributes.

        Parameters:
        rule_list: list of tuples representing the rule objects in order.
                    Each tuple is: (rule, alpha, n_pos_captured, n_neg_captured).
                    The last rule should be the leaf (where we store the loss).
        loss: numeric value representing the loss (or objective) for the complete rule list,
                which will be stored at the leaf.
        """
        current = self.root
        # Keep track of the nodes along the path for later updating.
        path = [self.root]
        for i, data in enumerate(rule_list):
            rule, alpha = data
            if rule not in current.children:
                # For the leaf node, store the loss.
                node_loss = loss if i == len(rule_list) - 1 else None
                new_node = TrieNode(rule=rule, probability=alpha, loss=node_loss)
                current.add_child(new_node)
            else:
                new_node = current.children[rule]
                # If we're at the last rule and the node is already present but wasn't marked as a leaf,
                # update its loss so that it represents a complete rule list.
                if i == len(rule_list) - 1 and new_node.loss is None:
                    new_node.loss = loss
            current = new_node
            path.append(current)

        # Now update subtree_count and max_height along the insertion path.
        # We update from the bottom (leaf) upward.
        for node in reversed(path):
            if not node.children:
                # If this node is a leaf (has no children)
                node.subtree_count = 1 if node.loss is not None else 0
                node.max_height = 0
            else:
                # Internal node: combine information from children.
                node.subtree_count = sum(child.subtree_count for child in node.children.values())
                node.max_height = 1 + max(child.max_height for child in node.children.values())
            node.visits += 1

    def remove(self, rule_list):
        """
        Removes a rule list from the trie, updating subtree_count and max_height accordingly.

        Parameters:
        rule_list: list of tuples representing the rule objects in order.
                    Each tuple is: (rule, alpha).
                    The last rule should correspond to a leaf node.
        """
        current = self.root
        path = [self.root]
        nodes_and_keys = []  # Track (parent_node, key) pairs

        for rule, _ in rule_list:
            if rule not in current.children:
                # Rule list not found; nothing to remove
                return
            nodes_and_keys.append((current, rule))
            current = current.children[rule]
            path.append(current)

        # Remove the leaf node
        parent, rule_key = nodes_and_keys[-1]
        if rule_key in parent.children:
            del parent.children[rule_key]

        # Now update metadata from the parent upwards
        for node in reversed(path):
            if not node.children:
                node.subtree_count = 1 if node.loss is not None else 0
                node.max_height = 0
            else:
                node.subtree_count = sum(child.subtree_count for child in node.children.values())
                node.max_height = 1 + max(child.max_height for child in node.children.values())

    def update_subtree_info(self, node=None):
        """
        Updates the subtree_count and max_height for every node in the trie.
        This is done via a post-order traversal.
        For a leaf (node with no children), subtree_count is 1 and max_height is 0.
        For an internal node, subtree_count is the sum of its children’s counts,
        and max_height is 1 plus the maximum child height.
        """
        if node is None:
            node = self.root

        # If node has no children, check if it's a valid leaf (i.e. a rule list ending with -1 or -2).
        if not node.children:
            # Only consider nodes representing complete rule lists (leaves with predictions).
            if node.rule in (-1, -2):
                node.subtree_count = 1
            else:
                node.subtree_count = 0
            node.max_height = 0
            return node.subtree_count, node.max_height

        total_count = 0
        max_child_height = 0
        for child in node.children.values():
            child_count, child_height = self.update_subtree_info(child)
            total_count += child_count
            if child_height > max_child_height:
                max_child_height = child_height

        node.subtree_count = total_count
        node.max_height = 1 + max_child_height
        return node.subtree_count, node.max_height

    def display(self, node=None, depth=0):
        """Recursively displays the trie structure."""
        if node is None:
            node = self.root
        indent = "  " * depth
        print(f"{indent}{node}")
        for child in node.children.values():
            self.display(child, depth + 1)

    def find_node(self, prefix, antecedent):
        """
        Traverse the trie following the sequence of rules provided in rule_list.
        
        Parameters:
            rule_list (list): A list of rules (e.g., [3, 5, -2]) to follow in order.
        
        Returns:
            TrieNode: The node corresponding to the end of the path if found, otherwise None.
        """
        current = self.root
        for rule, _ in prefix:
            if rule in current.children:
                current = current.children[rule]
            else:
                return None
        if antecedent in current.children:
            return current.children[antecedent]
        return None
