from copy import deepcopy
from graph import Tree, InternalNode, Leaf

class DecisionTreePruner:
    def __init__(self, tree: Tree, alpha: float = 0.01):
        self.tree = tree
        self.alpha = alpha
        
        self.best_pruned_tree = None
        self.best_prune_score = float('inf')
        self.pruned_trees = []
        self.counter = 0

    def prune(self):
        while True:
            number_of_nodes = len(self.tree.get_internal_nodes())
            self.pruned_trees.append(self.tree)
        
            for i in range(number_of_nodes):
                working_tree = deepcopy(self.tree)
                internal_nodes = working_tree.get_internal_nodes()
                node = internal_nodes[i]
                self._prune_node(node, tree=working_tree)
                self.pruned_trees.append(working_tree)
            
            for tree in self.pruned_trees:
                self.counter += 1
                score = self._compute_cost(tree)
                if score < self.best_prune_score:
                    self.best_prune_score = score
                    self.best_pruned_tree = tree
            
            # print("I am here", self.best_prune_score, self.best_pruned_tree)
            if not self.best_pruned_tree:
                break
            
            self.tree = self.best_pruned_tree
            self._reset()
        return self.tree
    
    def _reset(self):
        self.best_pruned_tree = None
        self.pruned_trees = []
        
    def _prune_node(self, node, tree=None):
        # Choose the majority class of the node according to the node.data
        new_leaf = None
        if not isinstance(node, InternalNode):
            print("Cannot prune the leaf node.")
            exit()
    
        dataframe = node.data
        labels_column = dataframe.iloc[:, -1]
        counts = labels_column.value_counts()
        max_label = counts.idxmax()

        new_leaf = Leaf(label=max_label,
                        positive=counts[max_label],
                        negative=counts.sum() - counts[max_label],
                        parent=node.parent,
                        data=dataframe)
        new_leaf.value = node.value

        if node.parent:
            node.parent.children.remove(node)
            node.parent.children.append(new_leaf)
        else:
            if tree:
                tree.root = new_leaf
            else:
                self.tree.root = new_leaf 
        
    def _compute_cost(self, tree):
        errors = self._get_number_of_errors(tree.root)
        return errors / len(tree.root.data) + 0.01 * tree.count_leaves()

    def _get_number_of_errors(self, node):
        if isinstance(node, Leaf):
            return node.negative
        else:
            errors = 0
            for child in node.children:
                errors += self._get_number_of_errors(child)
            return errors