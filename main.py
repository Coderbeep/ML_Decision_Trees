import pandas as pd
from pruner import DecisionTreePruner
from criteria import InformationGain, InformationGainRatio, GiniIndex, AttributeSelectionStrategy
from graph import Graph, Tree, InternalNode, Leaf, Node
from node_factory import NodeFactory

# TODO: labels as another pd series, passed as additional argument 
#       to the algorithm class
# TODO: refactor the ID3 Algorithm to handle the positive and negative labels inside
#       the node class
# TODO: backtracking in ID3 Algorithm 
# TODO: C4.5 Algorithm
#       - Discrete and continuous attributes handling
#       - Missing values handling  
#           - Ignore the instance
#           - Take into account only a fraction of the instance


# The "same" attribute can be chosen of the same level
# The C4.5 introduces thresholds, so the same attribute with different thresholds
# can be chosen at the same level


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
        self.node_factory = NodeFactory()

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

    def _algorithm_id3(self, data, attributes, labels_column, used_attributes=None):
        if not used_attributes:
            used_attributes = set()

        if len(data) == 0:
            return None

        if len(labels_column.unique()) == 1:
            return Leaf(label=labels_column.iloc[0])

        if len(attributes.columns) == 0:
            label_counts = labels_column.value_counts()
            max_count = label_counts.max()
            if len(label_counts[label_counts == max_count]) > 1:
                print("The dataset is inconsistent. Check records: " + str(data.index.tolist()))
                exit()
            return Leaf(label=label_counts.idxmax())

        available_attributes = set(attributes.columns) - used_attributes
        best_attribute, best_measure_value = self.criterion.calculate(data, available_attributes)
        if not best_attribute:
            return Leaf(label=labels_column.value_counts().idxmax())
                
        node = InternalNode(attribute=best_attribute)
        for value in data[best_attribute].unique():
            ids = data.index[data[best_attribute] == value].tolist()
            sub_data = data.loc[ids, :]
            sub_labels_column = sub_data.iloc[:, -1]
            sub_attributes = attributes.drop(best_attribute, axis=1)
            updated_used_attributes = used_attributes.union({best_attribute})
            child_node = self._algorithm_id3(sub_data, sub_attributes, sub_labels_column, used_attributes=updated_used_attributes)
            child_node.value = value
            node.add_child(child_node)
        return node
    
    def execute(self):
        self.tree = Tree(self._algorithm_id3(self.data, self.attributes, self.labels_column))
        return self.tree

    def render_graph(self):
        graph = Graph(self.tree, self.attribute_names, self.unique_labels)
        return graph
    
    def __str__(self):
        description = "ID3 Algorithm\n"
        description += "Unique labels: {}\n".format(str(self.unique_labels))
        description += "Attributes: {}\n".format(str(self.attribute_names))
 
""" C4 Algorithm = ID3 + cost-complexity pruning 
    Cost-complexity: minimize E/n + alpha * |T|, where 
        E is the number of misclassified instances, 
        n is the total number of instances, 
        alpha is a parameter,
        |T| is the number of leaves in the tree.
"""    

class C45AlgorithmCont(TreeAlgorithm):
    def __init__(self, 
                 data: pd.DataFrame, 
                 criterion: str = "inf_gain", 
                 labels_column: int = -1,
                 alpha: float = 0.01):
        super().__init__(data, criterion, labels_column)
        self.alpha = alpha
        
    def _get_number_of_errors(self, node) -> int:
        if isinstance(node, Leaf):
            self.leaves += 1
            self.errors += node.negative
        else:
            for child in node.children:
                self._get_number_of_errors(child)
        return self.errors
    
    def _algorithm_c45(self, data, attributes, labels_column, used_attributes=None, parent=None) -> Node:
        if not used_attributes:
            used_attributes = list()

        if len(data) == 0:
            return None
        
        if len(labels_column.unique()) == 1:
            return self.node_factory.create_leaf(labels_column, parent, data)

        if len(attributes.columns) == 0:
            self.node_factory.create_leaf(labels_column, parent, data)

        available_attributes = [x for x in attributes.columns if x not in used_attributes]
        best_attribute, best_measure_value, threshold = self.criterion.calculate(data, available_attributes)

        if not best_attribute:
            return self.node_factory.create_leaf(labels_column, parent, data)
                
        node = InternalNode(attribute=best_attribute, parent=parent, data=data)

        for x in [0, 1]:
            ids = data.index[data[best_attribute] <= threshold] if x == 0 else data.index[data[best_attribute] > threshold]
            sub_data = data.loc[ids, :]
            sub_labels_column = sub_data.iloc[:, -1]
            sub_attributes = attributes
            updated_used_attributes = used_attributes.append(best_attribute)
            child_node = self._algorithm_c45(sub_data, sub_attributes, sub_labels_column, used_attributes=updated_used_attributes, parent=node)
            child_node.value = f"<={threshold:.3f}" if x == 0 else f"> {threshold:.3f}"
            node.add_child(child_node)
        return node

    def execute(self) -> Tree:
        self.tree = Tree(self._algorithm_c45(self.data, self.attributes, self.labels_column))

        pruner = DecisionTreePruner(self.tree, self.alpha)
        pruned_tree = pruner.prune()
        
        return pruned_tree

import matplotlib.pyplot as plt

def visualize_error_with_alpha(data):
    alpha = 0.001
    errors = []
    c45_algorithm = C45AlgorithmCont(data, criterion="inf_gain", alpha=alpha)
    while alpha < 0.2:
        tree = c45_algorithm.execute()
        c45_algorithm.alpha = alpha
        errors.append(tree.get_number_of_errors())
        print(alpha, errors[-1])
        alpha += 0.0001
        
    plt.plot(errors)
    plt.show()

def test_c45_algorithm():
    data = pd.read_csv('data/iris.csv')
    c4_algorithm = C45AlgorithmCont(data, criterion="inf_gain", alpha=0)
    tree = c4_algorithm.execute()
    tree.visualize()

# TEST SETTINGS
# - DATASET: iris.csv
# - CRITERION: Information Gain
# - ALPHA: 0.08

# Initially ~ 2.4 seconds for 5 executions
# Vectorization in numpy ~ 0.7 seconds for 5 executions
# Entropy calc on labels only ~ 0.6 seconds for 5 executions
# Numpy on labels change ~ 0.5 seconds for 5 executions

if __name__ == "__main__":
    # execution_time = timeit.timeit(test_c45_algorithm, number=5)
    # print(f"Execution time: {execution_time:.3f} seconds")
    # visualize_error_with_alpha(pd.read_csv('data/iris.csv'))
    test_c45_algorithm()