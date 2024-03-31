from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd

data = pd.read_csv('data.csv')
classifier = DecisionTreeClassifier()

labels = data['Play?']
data = data.drop('Play?', axis=1)

classifier.fit(data, labels)


