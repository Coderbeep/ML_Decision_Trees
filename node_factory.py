from graph import Tree, InternalNode, Leaf, Node
import pandas as pd

class NodeFactory:
    def __init__(self):
        pass
    
    def create_leaf(self, 
                    labels: pd.Series, 
                    parent: Node, 
                    data: pd.DataFrame):
        # If all labels are the same, return a leaf node
        if len(labels.unique()) == 1:
            return Leaf(label=labels.iloc[0], 
                        positive=len(labels), 
                        negative=0, 
                        parent=parent,
                        data=data)
        
        # If there are no more attributes to split, return a leaf node
        counts = labels.value_counts()
        max_label = counts.idxmax()
        return Leaf(label=labels.value_counts().idxmax(),
                    positive=counts[max_label],
                    negative=counts.sum() - counts[max_label],
                    parent=parent,
                    data=data)