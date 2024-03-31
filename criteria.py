import pandas as pd
import numpy as np

class AttributeSelectionStrategy():
    def calculate(self, data, attributes):
        raise NotImplementedError
    
class InformationGain(AttributeSelectionStrategy):
    def calculate(self, data, attributes):
        best_attribute = None
        best_inf_gain = -1
        
        initial_entropy = self._get_entropy(data)
        for attribute in attributes:
            information_gain = initial_entropy - self._get_split_entropy(data, attribute)
            if information_gain > best_inf_gain:
                best_inf_gain = information_gain
                best_attribute = attribute
                
        return best_attribute, best_inf_gain

    def _get_entropy(self, data: pd.DataFrame):
        total_rows = len(data)
        # Count unique values in labels column according to the ids
        labels = data.iloc[:, -1]
        counts = labels.value_counts()
        
        # Calculate entropy
        probabilities = counts / total_rows
        entropy = -(probabilities * np.log2(probabilities)).sum()
        return entropy

    def _get_split_entropy(self, data: pd.DataFrame, attribute_name):
        total_rows = len(data)
        
        labels = data.iloc[:, -1]
        attributes = data.iloc[:][attribute_name]
        counts = pd.crosstab(attributes, labels)
        attribute_totals = counts.sum(axis=1)

        # Calculate split entropy
        attribute_probabilities = counts.div(attribute_totals, axis=0)
        attribute_probabilities.replace(0, np.finfo(float).eps, inplace=True)
        
        attribute_entropy = -(attribute_probabilities * np.log2(attribute_probabilities)).sum(axis=1)
        split_entropy = (attribute_totals / total_rows) * attribute_entropy
        return split_entropy.sum()
    
class InformationGainRatio(AttributeSelectionStrategy):
    def calculate(self, data, attributes):
        best_attribute = None
        best_inf_gain_ratio = -1
        
        initial_entropy = self._get_entropy(data)
        for attribute in attributes:
            information_gain = initial_entropy - self._get_split_entropy(data, attribute)
            attr_entropy = self._get_entropy_column(data[attribute])
            if attr_entropy == 0:
                continue
            information_gain_ratio = information_gain / attr_entropy
            if information_gain_ratio > best_inf_gain_ratio:
                best_inf_gain_ratio = information_gain_ratio
                best_attribute = attribute
                
        return best_attribute, best_inf_gain_ratio

    def _get_entropy_column(self, column: pd.Series):
        total_rows = len(column)
        counts = column.value_counts()
        
        # Calculate entropy
        probabilities = counts / total_rows
        entropy = -(probabilities * np.log2(probabilities)).sum()
        return entropy
    
    def _get_entropy(self, data: pd.DataFrame):
        total_rows = len(data)
        # Count unique values in labels column according to the ids
        labels = data.iloc[:, -1]
        counts = labels.value_counts()
        
        # Calculate entropy
        probabilities = counts / total_rows
        entropy = -(probabilities * np.log2(probabilities)).sum()
        return entropy

    def _get_split_entropy(self, data: pd.DataFrame, attribute_name):
        total_rows = len(data)
        
        labels = data.iloc[:, -1]
        attributes = data.iloc[:][attribute_name]
        counts = pd.crosstab(attributes, labels)
        attribute_totals = counts.sum(axis=1)

        # Calculate split entropy
        attribute_probabilities = counts.div(attribute_totals, axis=0)
        attribute_probabilities.replace(0, np.finfo(float).eps, inplace=True)
        
        attribute_entropy = -(attribute_probabilities * np.log2(attribute_probabilities)).sum(axis=1)
        split_entropy = (attribute_totals / total_rows) * attribute_entropy
        return split_entropy.sum()

class GiniIndex(AttributeSelectionStrategy):
    def calculate(self, data, attributes):
        best_attribute = None
        best_gini_index = 1
        
        for attribute in attributes:
            gini_index = self._get_gini_index(data, attribute)
            if gini_index < best_gini_index:
                best_gini_index = gini_index
                best_attribute = attribute
                
        return best_attribute, best_gini_index

    def _get_gini_index(self, data: pd.DataFrame, attribute_name):
        total_rows = len(data)
        
        labels = data.iloc[:, -1]
        attributes = data.iloc[:][attribute_name]
        counts = pd.crosstab(attributes, labels)
        attribute_totals = counts.sum(axis=1)

        # Calculate gini index
        attribute_probabilities = counts.div(attribute_totals, axis=0)
        attribute_probabilities.replace(0, np.finfo(float).eps, inplace=True)
        
        attribute_gini = 1 - (attribute_probabilities ** 2).sum(axis=1)
        gini_index = (attribute_totals / total_rows) * attribute_gini
        return gini_index.sum()
