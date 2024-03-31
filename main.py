import pandas as pd
from criteria import InformationGain
from graph import Graph


def read_data_csv(file_path):
    data = pd.read_csv(file_path)
    return data

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
    def __init__(self, data: pd.DataFrame, labels_column: int = -1):
        self.data = data
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
    def __init__(self, data: pd.DataFrame, labels_column: int = -1):
        super().__init__(data, labels_column)

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
        best_attribute, best_measure_value = InformationGain().calculate(data, available_attributes)
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
        return description
    
if __name__ == "__main__":
    data = read_data_csv('attractive.csv')
    id3 = ID3Algorithm(data, -1)
    tree = id3.execute()
    print(id3)
    
    id3.render_graph()
