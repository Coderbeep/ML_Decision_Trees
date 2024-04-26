from graph import Tree, Leaf
import numpy as np
import pandas as pd

class Rules():
    def __init__(self, tree: Tree) -> None:
        self.tree = tree
        self.rules = Rules.get_rules(tree)
    
    """ Iterates through leaf nodes and returns the indices of the misclassified instances 
    in the dataset.
    
    Args: 
        tree (Tree): The decision tree
        
    Returns:
        np.array: The indices of the misclassified instances in the dataset """
    def get_wrong_indices(self) -> np.ndarray:
        leaves = self.tree.get_leaves()
        wrong_indices = []
        
        for leaf in leaves:
            correct_label = leaf.label
            wrong = leaf.data[leaf.data.iloc[:, -1] != correct_label].index
            wrong_indices.extend(wrong)

        return wrong.values
    
    def get_ruled_dataset(self, initial_data: pd.DataFrame) -> pd.DataFrame:
        atomic_rules = Rules.get_atomic_rules(self.tree)
        
        ruled_data = pd.DataFrame(columns=atomic_rules)
        
        for rule in atomic_rules:
            attr, operator, value = Rules.parse_rule(rule)
            value = bool(1) if value == "True" else bool(0) if value == "False" else value
            if operator == "=":
                ruled_data[rule] = (initial_data[attr] == value).astype(int)
            elif operator == "<":
                ruled_data[rule] = (initial_data[attr] < value).astype(int)
            elif operator == "<=":
                ruled_data[rule] = (initial_data[attr] <= value).astype(int)
            elif operator == ">":
                ruled_data[rule] = (initial_data[attr] > value).astype(int)
            elif operator == ">=":
                ruled_data[rule] = (initial_data[attr] >= value).astype(int)
        
        ruled_data['label'] = initial_data.iloc[:, -1]
        
        return ruled_data
            
    @staticmethod
    def get_rules(tree: Tree) -> list:
        rules = list()
        def traverse(node, rule):
            if isinstance(node, Leaf):
                rules.append(rule.split())
            else:
                for child in node.children:
                    new_rule = rule + f"{node.attribute}_{child.value} "
                    traverse(child, new_rule)
        traverse(tree.root, "")
        return rules

    @staticmethod
    def get_atomic_rules(tree: Tree) -> list:
        rules = set()
        def traverse(node):
            if not isinstance(node, Leaf):
                for child in node.children:
                    rules.add(f"{node.attribute}_{child.value}")  
                    traverse(child)
        traverse(tree.root)
        return sorted(list(rules))
    
    @staticmethod
    def parse_rule2(rule: list) -> str:
        return " & ".join([str(node) for node in rule])
    
    @staticmethod
    def parse_rule(rule):
        attribute, condition = rule.split("_")
        if condition.startswith("<="):
            return attribute, "<=", float(condition[2:])
        elif condition.startswith("<"):
            return attribute, "<", float(condition[1:])
        elif condition.startswith(">="):
            return attribute, ">=", float(condition[2:])
        elif condition.startswith(">"):
            return attribute, ">", float(condition[1:])
        else:
            return attribute, "=", condition
        
if __name__ == "__main__":
    pass