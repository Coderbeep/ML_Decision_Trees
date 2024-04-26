import matplotlib.pyplot as plt
import torch

from torch import nn
from rules import Rules
from main import test_c45_algorithm
from graph import Tree

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" The data is assumed to be prepared for classification """
class DataPreprocessor():
    def __init__(self, tree: Tree):
        self.tree = tree
        self.data = tree.root.data
        self.translation = {"Setosa": 0, "Versicolor": 1, "Virginica": 2}
        
        self.rules = Rules(tree)
        
        self.preprocess()
        
    def preprocess(self):
        # rename the variety to label
        self.data = self.data.rename(columns={"variety": "label"})
        
        self.data["label"] = self.data["label"].map(self.translation)
        return self.data
    
    def generate_mask(self):
        atomic_rules = Rules.get_atomic_rules(self.tree)
        rules = Rules.get_rules(self.tree)
        indices = torch.zeros(len(atomic_rules), len(rules), dtype=torch.long)
        for i, atomic_rule in enumerate(atomic_rules):
            for j, rule in enumerate(rules):
                if atomic_rule in rule:
                    indices[i][j] = 1
        return indices.float()
    
    def get_full_data(self):
        data = self.rules.get_ruled_dataset(self.data)
        return data
    
    def get_correct_data(self):
        data = self.rules.get_ruled_dataset(self.data)
        
        wrong_indices = Rules(self.tree).get_wrong_indices()
        data = data.drop(wrong_indices)
        return data
    


preprocessor = DataPreprocessor(test_c45_algorithm())

IN_FEATURES = len(Rules.get_atomic_rules(preprocessor.tree))
OUT_FEATURES = 3
HIDDEN_NODES = len(Rules.get_rules(preprocessor.tree))

# define a mask for the weights
class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        self.fc1 = nn.Linear(IN_FEATURES, HIDDEN_NODES)
        self.fc2 = nn.Linear(HIDDEN_NODES, OUT_FEATURES)
        self.mask = nn.Parameter(preprocessor.generate_mask(), requires_grad=False)
        
    def forward(self, x):
        self.fc1.weight.data = self.fc1.weight * self.mask.T
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        return x
    
model = CustomNet().to(device)

df_correct = preprocessor.get_correct_data()

print(df_correct.head(10))

X = torch.tensor(df_correct.iloc[:, :-1].values, dtype=torch.int64).to(device)
y = torch.tensor(df_correct.iloc[:, -1].values, dtype=torch.int64).to(device)

# define the loss function
criterion = nn.CrossEntropyLoss()
# define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)



# train the model
losses = []
for epoch in range(1000):
    optimizer.zero_grad()
    output = model(X.float())
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")


df_full = preprocessor.get_full_data()

X = torch.tensor(df_full.iloc[:, :-1].values, dtype=torch.int64).to(device)
y = torch.tensor(df_full.iloc[:, -1].values, dtype=torch.int64).to(device)

# predict some values
model.eval()
with torch.no_grad():
    output = model(X.float())
    _, predicted = torch.max(output, 1)
    print(predicted)
    print(y)
    print((predicted == y).sum().item() / y.size(0))


