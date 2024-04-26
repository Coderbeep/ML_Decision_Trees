import os
import torch
from torch import nn
from main import get_rules

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# the number of rules is the number of leaves
# the amount of hidden nodes is the number of rules
# the input dimension is the number of features (number of smallest possible parts of rules)
# the atomic rules are supposed to be connected to the hidden layer nodes (but only to those that have such atomic rule in the whole rule)
# each hidden node is supposed to represent a rule (the whole one, not atomic)

df, rules, atomic_rules = get_rules()
atomic_rules = sorted(atomic_rules)

""" The function returns the indices of the hidden nodes,
    that are connected to the atomic rules (the smallest parts of the rules)
    which are used as input. The indices would be indicating where weight is 1 in the mask."""
def generate_indices(rules, atomic_rules):
    indices = torch.zeros(len(atomic_rules), len(rules), dtype=torch.long)
    for i, atomic_rule in enumerate(atomic_rules):
        for j, rule in enumerate(rules):
            if atomic_rule in rule:
                indices[i][j] = 1
    return indices

indices = generate_indices(rules, atomic_rules)

df_full = df.copy()

IN_FEATURES = 10
OUT_FEATURES = 3
HIDDEN_NODES = 7
# drop those rows from the dataframe
df = df.drop([77, 83])


# define a mask for the weights
class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        self.fc1 = nn.Linear(IN_FEATURES, HIDDEN_NODES)
        self.fc2 = nn.Linear(HIDDEN_NODES, OUT_FEATURES)
        self.mask = nn.Parameter(indices.float(), requires_grad=False)
        
    def forward(self, x):
        self.fc1.weight.data = self.fc1.weight * self.mask.T
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        return x
    
model = CustomNet().to(device)

# the df contains only ones and zeros (according to the input features)
# the target three classes are 0, 1, 2 (according to the iris labels in the dataset)
# the target is the last column in the dataframe

translation = {"Setosa": 0, "Versicolor": 1, "Virginica": 2}
df["label"] = df["label"].map(translation)

X = torch.tensor(df.iloc[:, :-1].values, dtype=torch.int64).to(device)
y = torch.tensor(df.iloc[:, -1].values, dtype=torch.int64).to(device)

# define the loss function
criterion = nn.CrossEntropyLoss()
# define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

import matplotlib.pyplot as plt

# train the model
losses = []
for epoch in range(1500):
    optimizer.zero_grad()
    output = model(X.float())
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

translation = {"Setosa": 0, "Versicolor": 1, "Virginica": 2}
df_full["label"] = df_full["label"].map(translation)

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


