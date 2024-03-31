from graphviz import Digraph

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
