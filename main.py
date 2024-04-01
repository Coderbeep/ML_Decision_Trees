import pandas as pd

from criteria import InformationGain, InformationGainRatio, GiniIndex, AttributeSelectionStrategy
from graph import Graph

criteria = {
    "inf_gain": InformationGain,
    "inf_gain_ratio": InformationGainRatio,
    "gini": GiniIndex
}

""" Interface for the tree algorithm.  

    Attributes:
        data (pd.DataFrame): The dataset
        labels_column_name (str): The name of the labels column
        labels_column (pd.Series): The labels column
        unique_labels (np.array): The unique labels
        attributes (pd.DataFrame): The attributes
        tree (dict): The decision tree
        attribute_names (set): The attribute names
    
    Methods:
        execute: Execute the algorithm
        render_graph: Render the decision tree
        __str__: String representation of the algorithm
"""
class TreeAlgorithm:
    def __init__(self, 
                 data: pd.DataFrame, 
                 criterion: str = "inf_gain", 
                 labels_column: int = -1):
        self.data = data
        try:
            self.criterion = criteria.get(criterion)()
        except:
            print(f"Invalid criterion. Try to use one of the following: {list(criteria.keys())}") 
            exit()

        self.labels_column = data.iloc[:, labels_column]
        self.unique_labels = self.labels_column.unique()
        self.attributes = self.data.drop(data.columns[labels_column], axis=1)

        self.tree = None
        self.attribute_names = set(self.attributes.columns)

    def execute(self):
        raise NotImplementedError

    def render_graph(self):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError

""" Available measures:
    - Information Gain
    - Information Gain Ratio
    - Gini Index """
class ID3Algorithm(TreeAlgorithm):
    def __init__(self, data: pd.DataFrame, criterion = "inf_gain", labels_column: int = -1):
        super().__init__(data, criterion, labels_column)

    def _algorithm_id3(self, data, attributes, labels_column, tree=None, used_attributes=None):
        ### The stopping conditions ###
        if not used_attributes:
            used_attributes = set()

        if len(data) == 0:
            return None

        if len(labels_column.unique()) == 1:
            return labels_column.unique()[0]

        if len(attributes.columns) == 0:
            return labels_column.value_counts().idxmax()

        available_attributes = set(attributes.columns) - used_attributes
        best_attribute, best_measure_value = self.criterion.calculate(data, available_attributes)
        if not best_attribute:
            return labels_column.value_counts().idxmax()
                
        if tree is None:
            tree = {}
        
        tree[best_attribute] = {}
        for value in data[best_attribute].unique():
            ids = data.index[data[best_attribute] == value].tolist()
            sub_data = data.loc[ids, :]
            labels_column = sub_data.iloc[:, -1]
            sub_attributes = attributes.drop(best_attribute, axis=1)
            updated_used_attributes = used_attributes.union({best_attribute})
            subtree = self._algorithm_id3(sub_data, sub_attributes, labels_column, tree={}, used_attributes=updated_used_attributes)        
            tree[best_attribute][value] = subtree
        return tree
    
    def execute(self):
        self.tree = self._algorithm_id3(self.data, self.attributes, self.labels_column)
        return self.tree

    def render_graph(self):
        graph = Graph(self.tree, self.attribute_names, self.unique_labels)
        return graph
    
    def __str__(self):
        description = "ID3 Algorithm\n"
        description += "Unique labels: {}\n".format(str(self.unique_labels))
        description += "Attributes: {}\n".format(str(self.attribute_names))


class LeafNode():
    def __init__(self, label):
        self.label = label 
        self.value = None

    def __str__(self, level=0):
        return f"{self.value} {self.label}"

    def __repr__(self):
        return f"{self.label}"
    
class InternalNode():
    def __init__(self, attribute):
        self.attribute = attribute
        self.value = None
        self.children = []
        
    def add_child(self, child):
        self.children.append(child)

    def __str__(self, level=0):
        return f"{self.attribute}, {self.value}, {self.children}"

    def __repr__(self):
        return f"{self.attribute}: {self.value} -> {self.children}"

class Tree:
    def __init__(self, root=None):
        self.root = root
        self.leaves = None

    def traverse_dfs(self, node=None):
        if node is None:
            node = self.root
        print("Here -> ", node)
        if isinstance(node, InternalNode):
            for child in node.children:
                self.traverse_dfs(child)

    def traverse_bfs(self):
        if self.root is None:
            return

        queue = [self.root]
        while queue:
            node = queue.pop(0)
            print("Here -> ", node)
            if isinstance(node, InternalNode):
                for child in node.children:
                    queue.append(child)
    
    def count_leaves(self, node=None):
        if self.leaves is not None:
            return self.leaves
        
        if node is None:
            node = self.root

        if isinstance(node, LeafNode):
            return 1

        count = 0
        for child in node.children:
            count += self.count_leaves(child)
        return count
 
""" C4 Algorithm = ID3 + cost-complexity pruning 
    Cost-complexity: minimize E/n + alpha * |T|, where 
        E is the number of misclassified instances, 
        n is the total number of instances, 
        alpha is a parameter,
        |T| is the number of leaves in the tree.
"""    
class C4Algorithm(TreeAlgorithm):
    def __init__(self, data: pd.DataFrame, criterion: str = "inf_gain", labels_column: int = -1):
        super().__init__(data, criterion, labels_column)
        self.n = len(data)
        self.alpha = 0.01
        
        self.leaves = 0
        self.errors = 0  
    
    def _algorithm_c4(self, data, attributes, labels_column, used_attributes=None):
        if not used_attributes:
            used_attributes = set()

        if len(data) == 0:
            return None

        if len(labels_column.unique()) == 1:
            return LeafNode(label=labels_column.iloc[0])

        if len(attributes.columns) == 0:
            return LeafNode(label=labels_column.value_counts().idxmax())

        available_attributes = set(attributes.columns) - used_attributes
        best_attribute, best_measure_value = self.criterion.calculate(data, available_attributes)
        if not best_attribute:
            return LeafNode(label=labels_column.value_counts().idxmax())
                
        node = InternalNode(attribute=best_attribute)
        for value in data[best_attribute].unique():
            ids = data.index[data[best_attribute] == value].tolist()
            sub_data = data.loc[ids, :]
            sub_labels_column = sub_data.iloc[:, -1]
            sub_attributes = attributes.drop(best_attribute, axis=1)
            updated_used_attributes = used_attributes.union({best_attribute})
            child_node = self._algorithm_c4(sub_data, sub_attributes, sub_labels_column, used_attributes=updated_used_attributes)
            child_node.value = value
            node.add_child(child_node)
        return node
    
    def execute(self):
        self.tree = self._algorithm_c4(self.data, self.attributes, self.labels_column)
        return self.tree

    
if __name__ == "__main__":
    data = pd.read_csv('data/attractive.csv')
    id3 = C4Algorithm(data, criterion="inf_gain")
    tree = id3.execute()
    tree_structure = Tree(tree)
    tree_structure.traverse_dfs()
