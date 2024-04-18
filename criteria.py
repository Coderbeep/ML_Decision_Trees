import pandas as pd
import numpy as np
""" The AttributeSelectionStrategy interface.
    
    Methods:
        calculate: Calculate the best attribute to split the data on
        
    Available strategies:
        - InformationGain
        - InformationGainRatio
        - GiniIndex
"""
class AttributeSelectionStrategy():
    def calculate(self, data: pd.DataFrame, attributes: set) -> tuple[str, float]:
        raise NotImplementedError

class InformationGain(AttributeSelectionStrategy):
    # def calculate(self, data: pd.DataFrame, attributes: set) -> tuple[str, float]:
    #     best_attribute = None
    #     best_inf_gain = 0
        
    #     initial_entropy = self._get_entropy(data)
    #     for attribute in attributes:
    #         information_gain = initial_entropy - self._get_split_entropy(data, attribute)
    #         if information_gain > best_inf_gain:
    #             best_inf_gain = information_gain
    #             best_attribute = attribute
            
    #     return best_attribute, best_inf_gain
    
    def calculate(self, data: pd.DataFrame, attributes: set) -> tuple[str, float]:
        best_attribute = None
        best_inf_gain = 0
        best_threshold = None
        
        for attribute in attributes:
            inf_gain = 0
            threshold = None
            
            if data[attribute].dtype == 'float64' or data[attribute].dtype == 'int64':
                cont_best_attribute, cont_inf_gain, cont_threshold = self._calculate_continuous_attribute(data, attribute)
                inf_gain = cont_inf_gain
                threshold = cont_threshold
            else:
                dis_best_attribute, dis_best_inf_gain, _ = self._calculate_discrete_attribute(data, attribute)
                inf_gain = dis_best_inf_gain
                threshold = None
                
            if inf_gain > best_inf_gain:
                best_inf_gain = inf_gain
                best_attribute = attribute
                best_threshold = threshold
                
        if data[best_attribute].dtype == 'float64' or data[best_attribute].dtype == 'int64':
            return best_attribute, best_inf_gain, best_threshold

        return best_attribute, best_inf_gain, None
    
    # Calculate helper which checks the datatype of the attribute
    # if float or integer, then calculate the best possible threshold
    # if not, then calculate the best possible split
    
    def _calculate_discrete_attribute(self, data: pd.DataFrame, attribute_name: str):
        best_attribute = None
        best_inf_gain = 0
        
        initial_entropy = self._get_entropy(data)
        information_gain = initial_entropy - self._get_split_entropy(data, attribute_name)
        if information_gain > best_inf_gain:
            best_inf_gain = information_gain
            best_attribute = attribute_name
        
        return best_attribute, best_inf_gain, None
    
    
    def _calculate_continuous_attribute(self, data: pd.DataFrame, attribute_name: str):
        best_threshold = None
        best_inf_gain = 0
        sorted_data = data.sort_values(by=attribute_name).reset_index(drop=True)
        n_instances = sorted_data.shape[0]
        
        overall_entropy = self._get_entropy(data)
        where_labels_change = self._get_places_where_labels_change(data, attribute_name)
        for i in range(where_labels_change.size):
            data_left = sorted_data.iloc[:where_labels_change[i]]
            data_right = sorted_data.iloc[where_labels_change[i]:]
            
            entropy_left = (data_left.shape[0] / n_instances) * self._get_entropy(data_left)
            entropy_right = (data_right.shape[0] / n_instances) * self._get_entropy(data_right)

            information_gain = overall_entropy - (entropy_left + entropy_right)
            if information_gain > best_inf_gain:
                best_inf_gain = information_gain
                best_threshold = (sorted_data.iloc[where_labels_change[i]][attribute_name] + sorted_data.iloc[where_labels_change[i] - 1][attribute_name]) / 2
            # print(f"Threshold: {(sorted_data.iloc[where_labels_change[i]][attribute_name] + sorted_data.iloc[where_labels_change[i] - 1][attribute_name]) / 2}")
        return attribute_name, best_inf_gain, best_threshold
    
    """ Returns the indexes where the labels change in the sorted (ascending) order 
        according to the attribute_name column. 
        
        The label is assumed to be the last column in the DataFrame."""
    def _get_places_where_labels_change(self, data: pd.DataFrame, attribute_name: str):
        sorted_data = data.sort_values(by=attribute_name)
        sorted_data.reset_index(drop=True, inplace=True)
        where_data_changed = (sorted_data.iloc[:, -1] != sorted_data.iloc[:, -1].shift())
        where_data_changed[0] = False
        
        return where_data_changed[where_data_changed == True].index.values

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
