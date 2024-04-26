from graphviz import Digraph
import uuid
import numpy as np

    # Issues That Affect ID3
    # Bad data: Two identical attributes given different results.
    # Missing data: Perhaps its too difficult to obtain / skip N/A.
    # Continuous Attribute Sets: Income is a real number.
    
class Graph:
    def __init__(self, tree, attributes, labels):
        self.tree = tree
        self.graph = Digraph("id3_tree")
        self.counter = {}
        self.attributes = attributes
        self.labels = labels
        
        self.preprocess_tree(self.tree)
        self.create_graph(self.tree, self.graph)
        self.graph.render('id3-tree', directory='output', format='png', cleanup=True)

    def preprocess_tree(self, tree):
        for key, value in list(tree.items()):
            del tree[key]
            key = str(key)
            tree[key] = value
            if isinstance(value, dict):
                self.preprocess_tree(value)
            else:
                tree[key] = str(value)

    def create_graph(self, tree, graph, counter={}, parent=None, current=None):        
        for key, value in tree.items():
            if isinstance(value, dict): # it is a split
                node_label1 = None
                if key in self.attributes:
                    if key not in counter:
                        counter[key] = 1
                    else: 
                        counter[key] += 1
                    node_label1 = f"{counter[key]}_{key}"
                    self.attributes.add(node_label1)
                    graph.node(node_label1, key, shape='box')
                    
                if parent in self.attributes:
                    if node_label1 is None:
                        node_label1 = key
                    
                    graph.edge(parent, node_label1, label=current) 
                if node_label1 is None:
                    node_label1 = key
                self.create_graph(value, graph, counter, current, node_label1)
            else: # it is going to leaf
                if value in self.labels:
                    if value not in counter:
                        counter[value] = 1
                    else:
                        counter[value] += 1
                    node_label = f"{counter[value]}_{value}" 
                    graph.node(node_label, value, shape='ellipse')
                    if current is not None:
                        graph.edge(current, node_label, label=key)  # Create edge from current to node_label
        return graph

class Node():
    def __init__(self, parent = None, data = None):
        self.uuid = str(uuid.uuid4())
        self.parent: Node = None
        self.data = data
        
class Leaf(Node):
    def __init__(self, label, positive=0, negative=0, parent=None, data = None):
        super().__init__(parent, data) 
        self.label = label 
        self.value = None
        
        self.positive: int = positive
        self.negative: int = negative
        
    def get_impurity(self):
        # calculate the impurity based on entropy
        labels = self.data.iloc[:, -1]
        counts = labels.value_counts()
        total_rows = len(self.data)
        probabilities = counts / total_rows
        entropy = -(probabilities * np.log2(probabilities)).sum()
        return entropy
    
    def get_wrongly_classified(self):
        pass
        
    def __str__(self, level=0):
        return f"{self.value} {self.label}"

    def __repr__(self):
        return f"{self.label}"
    
class InternalNode(Node):
    def __init__(self, attribute, parent=None, data = None):
        super().__init__(parent, data) 
        self.attribute = attribute
        self.value = None
        self.children = []
        # print(f"Creating node: {self.uuid}")

    def add_child(self, child):
        child.parent = self
        self.children.append(child)
        
    def get_leaves(self):
        leaves = list()
        def traverse(node):
            if isinstance(node, Leaf):
                leaves.append(node)
            else:
                for child in node.children:
                    traverse(child)
        traverse(self)
        return leaves
        
    def get_impurity(self):
        # TODO: Use Strategy Pattern to calculate impurity
        # calculate the impurity based on entropy
        labels = self.data.iloc[:, -1]
        counts = labels.value_counts()
        total_rows = len(self.data)
        probabilities = counts / total_rows
        entropy = -(probabilities * np.log2(probabilities)).sum()
        return entropy
    
        

    def __str__(self, level=0):
        return f"{self.attribute}, {self.value}, {self.children}"

    def __repr__(self):
        return f"{self.attribute}: {self.value} -> {self.children}"

class Tree:
    def __init__(self, root=None):
        self.root = root
        self.leaves = None

    def get_internal_nodes(self) -> list[Node]:
        nodes = list()
        def traverse(node):
            if isinstance(node, InternalNode):
                nodes.append(node)
                for child in node.children:
                    traverse(child)
        traverse(self.root)
        return nodes
    
    def get_leaves(self) -> list[Node]:
        if self.leaves is None:
            self.leaves = list()
        def traverse(node):
            if isinstance(node, Leaf):
                self.leaves.append(node)
            else:
                for child in node.children:
                    traverse(child)
        traverse(self.root)
        return self.leaves
    
    def get_number_of_nodes(self):
        return len(self.get_internal_nodes()) + self.count_leaves()

    def traverse_dfs(self, node=None):
        if node is None:
            node = self.root
        if isinstance(node, InternalNode):
            for child in node.children:
                self.traverse_dfs(child)

    def traverse_bfs(self):
        if self.root is None:
            return

        queue = [self.root]
        while queue:
            node = queue.pop(0)
            if isinstance(node, InternalNode):
                for child in node.children:
                    queue.append(child)
    
    def count_leaves(self, node=None):
        if node is None:
            node = self.root

        if isinstance(node, Leaf):
            return 1

        count = 0
        for child in node.children:
            count += self.count_leaves(child)
        return count

    def get_number_of_errors(self):
        errors = 0
        leaves = self.get_leaves()
        for leaf in leaves:
            errors += leaf.negative
        return errors
        
    def visualize(self, filename="result"):
        dot = Digraph(comment='Decision Tree')
        
        def add_nodes_edges(node, dot=None):
            if dot is None:
                dot = Digraph()
                dot.node(name=str(node.attribute), label=str(node.attribute))
                        
            if isinstance(node, InternalNode):
                dot.node(name=node.uuid, label=str(node.attribute), shape='box')
                for child in node.children:
                    if isinstance(child, Leaf):
                        dot.node(name=child.uuid, label=str(child.label) + f" {child.positive}/{child.negative}")
                        dot.edge(str(node.uuid), str(child.uuid), label=str(child.value))
                        # print(f"Creating leaf: {str(child.label)}")
                    if isinstance(child, InternalNode):
                        dot.node(name=child.uuid, label=str(child.attribute), shape='box')
                        dot.edge(str(node.uuid), str(child.uuid), label=str(child.value))
                        # print(f"Creating node: {str(child.attribute)}")
                    add_nodes_edges(child, dot)
            # the case for graph being only the root
            if isinstance(node, Leaf):
                dot.node(name=node.uuid, label=str(node.label) + f" {node.positive}/{node.negative}")
            
            return dot

        dot = add_nodes_edges(self.root, dot)
        dot.render(filename, format='png', cleanup=True)