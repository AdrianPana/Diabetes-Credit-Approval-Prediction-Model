from __future__ import annotations
from typing import Callable
from copy import deepcopy
from collections import Counter

import pandas as pd
import numpy as np
from graphviz import Digraph
from matplotlib import pyplot as plt
from matplotlib import image as mpimg

from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

from utils import prepare_data, plot_confusion_matrix, get_best_split_value, get_metrics

MIN_SAMPLES_PER_NODE = 2
MAX_DEPTH = 3

def scikit_random_forest(df_train, numeric, categorical, target, labels, df_test, class_weights):
    max_depth = None
    min_samples_leaf = 1
    criterion = "gini"
    class_weight = class_weights # after testing, I have decided not to use it
    n_estimators = 8
    max_samples = None
    max_features = 'log2'


    forest = RandomForestClassifier(criterion=criterion, min_samples_leaf=min_samples_leaf, n_estimators=n_estimators,
                                     bootstrap=True, max_samples=max_samples,
                                     max_features=max_features)
    
    X_train, y_train, X_test, y_test = prepare_data(df_train, numeric, categorical, target, df_test)

    forest.fit(X_train, y_train)

    predictions = forest.predict(X_test)

    plot_confusion_matrix(predictions, y_test)

    return get_metrics(y_test, predictions, labels)

#######################################################################################

class DecisionTreeNode:

    def __init__(self, 
                 feature = None,
                 children = None, 
                 label = None,
                 labels = None):

        self.split_feature = feature
        self.children = children if (children is not None and feature is not None) else {}
        self.label = label 
        self.labels = labels
        self.depth = 1        
        self.score = 0        
        self.num_samples = 0  
    
    def get_tree_graph(self,
                       graph = None):

        if graph is None:
            graph = Digraph()
            graph.attr('node', shape='box')
    
        if self.split_feature is None:
            fillcolor = 'darkolivegreen2' if self.label == self.labels[0] else 'yellow' if self.label == self.labels[1] else 'red'
            graph.node(f"{self}", f"Label: {self.label}\n"
                                  f"Score: {self.score:.3f}\n"
                                  f"Samples: {self.num_samples}", 
                       fillcolor=fillcolor, style='filled')
        else:
            graph.node(f"{self}", f"Split: {self.split_feature}?\n"
                                  f"Score: {self.score:.3f}\n"
                                  f"Samples: {self.num_samples}", fillcolor='lightblue', style='filled')
            
            for value, child in self.children.items():
                child.get_tree_graph(graph)
                graph.edge(f"{self}", f"{child}", label=f"{value}")
    
        return graph
    
    def display(self):
        graph = self.get_tree_graph()
        graph_path = '/tmp/tree_graph'
        graph.render(graph_path, format='png')
        
        img = mpimg.imread(f"{graph_path}.png")
        plt.figure(figsize=(100, 100))
        plt.imshow(img)
        plt.axis('off')
        plt.show()

################

class DecisionTree:

    def __init__(self,
                 split_strategy: str = 'random',
                 max_depth: int = np.inf,
                 min_samples_per_node: int = 1):

        self._root: DecisionTreeNode | None = None
        self._split_strategy: str = split_strategy
        self._max_depth: int = max_depth
        self._min_samples_per_node: int = min_samples_per_node
        
        
    @staticmethod
    def most_frequent_class(y: pd.Series) -> str:
        return y.value_counts().idxmax()
    
    
    @staticmethod
    def compute_entropy(y: pd.Series) -> float:

        value_counts = y.value_counts()
        total_count = y.count()
        entropy = 0
        for _, value in value_counts.items():
            entropy -= value / total_count * np.log2(value / total_count)

        return entropy
        
    
    @staticmethod
    def compute_information_gain(X: pd.DataFrame, y: pd.Series, feature: str) -> float:
        
        entropy_init = DecisionTree.compute_entropy(y)
        entropy_final = 0
        feature_values = X[feature].unique()

        for value in feature_values:
            feature_examples = X[X[feature] == value]
            feature_y = y[X[feature] == value]
            feature_entropy = DecisionTree.compute_entropy(feature_y)
            weight = feature_y.count() / y.count()
            entropy_final += weight * feature_entropy
        return entropy_init - entropy_final
    
    
    def _select_random_split_feature(self, X: pd.DataFrame, y: pd.Series, attribute_list: list[str]) -> str:
        return np.random.choice(attribute_list)
    
    
    def _select_best_split_feature(self, X: pd.DataFrame, y: pd.Series, attribute_list: list[str]) -> str:
        max = None
        max_att = None
        for att in attribute_list:
            gain = DecisionTree.compute_information_gain(X, y, att)
            if max == None or gain > max:
                max = gain
                max_att = att

        return max_att
    
    
    def _generate_tree(self,
                       parent_node: DecisionTreeNode | None,
                       X: pd.DataFrame,
                       y: pd.Series,
                       feature_list: list[str],
                       select_feature_func: Callable[[pd.DataFrame, pd.Series, list[str]], str],
                       labels) -> DecisionTreeNode:

        feature_list = deepcopy(feature_list)
        
        node = DecisionTreeNode(labels=labels)
        node.depth = parent_node.depth + 1 if parent_node is not None else 0
        node.score = DecisionTree.compute_entropy(y)  
        node.num_samples = len(y)
        node.label = DecisionTree.most_frequent_class(y)
        
        isLeaf = False
        if len(feature_list) == 0 or node.depth == self._max_depth or y.count() < self._min_samples_per_node or y.nunique() == 1:
            isLeaf = True
        
        if isLeaf:
            return node
        
        att = select_feature_func(X, y, feature_list)
        att_values = X[att].unique()
        if len(att_values) <= 7:
            node.split_feature = att
            feature_list.remove(att)
            for value in att_values:
                new_child = self._generate_tree(node,
                                                X[X[att] == value],
                                                y[X[att] == value],
                                                feature_list,
                                                select_feature_func,
                                                labels=labels)
                node.children[value] = new_child
        else:
            threshold = get_best_split_value(X, att, att_values, y)
            node.split_feature = f"{att} <= {threshold:.2f}"
            node.children[True] = self._generate_tree(node,
                                                      X[X[att] <= threshold],
                                                      y[X[att] <= threshold],
                                                      feature_list,
                                                      select_feature_func,
                                                      labels=labels)
            node.children[False] = self._generate_tree(node,
                                                      X[X[att] > threshold],
                                                      y[X[att] > threshold],
                                                      feature_list,
                                                      select_feature_func,
                                                      labels=labels)

        return node
    
        
    def fit(self, X: pd.DataFrame, y: pd.Series, labels):
        if self._split_strategy == 'random':
            select_feature_func = self._select_random_split_feature
        elif self._split_strategy == 'id3':
            select_feature_func = self._select_best_split_feature
        else:
            raise ValueError(f"Unknown split strategy {self._split_strategy}")
        
        self._root = self._generate_tree(parent_node=None,
                                         X=X,
                                         y=y,
                                         feature_list=X.columns.tolist(),
                                         select_feature_func=select_feature_func,
                                         labels=labels)
        
    def _predict_once(self, x: pd.Series) -> str:
        node = self._root
        
        while node.split_feature is not None:
            if node.split_feature in x and x[node.split_feature] in node.children:
                node = node.children[x[node.split_feature]]
            else:
                break
        return node.label
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.array([self._predict_once(x) for _, x in X.iterrows()])
    
    def get_depth(self) -> int:
        def _get_depth(node: DecisionTreeNode) -> int:
            if node is None:
                return 0
            return max([_get_depth(child) for child in node.children.values()], default=0) + 1
        
        return _get_depth(self._root)
    
    def get_number_of_nodes(self) -> int:
        def _get_number_of_nodes(node: DecisionTreeNode) -> int:
            if node is None:
                return 0
            return sum([_get_number_of_nodes(child) for child in node.children.values()], 0) + 1
        
        return _get_number_of_nodes(self._root)
    
    def get_tree_graph(self):
        return self._root.get_tree_graph()
    
    def display(self):
        return self._root.display()

#########################################################

def precision(y_pred: pd.Series, y_true: pd.Series, c: str) -> float:
    
    pred_c = y_pred[y_pred == c]
    
    true_c = y_true[(y_pred == c) & (y_true == c)]
    
    if len(pred_c) == 0:
        return 0
    else:
        return len(true_c) / len(pred_c)
    
def recall(y_pred: pd.Series, y_true: pd.Series, c: str) -> float:
    
    true_c = y_true[y_true == c]
    
    pred_c = y_pred[(y_pred == c) & (y_true == c)]
    
    if len(true_c) == 0:
        return 0
    else:
        return len(pred_c) / len(true_c)
    
def f1_score(y_pred: pd.Series, y_true: pd.Series, c: str) -> float:
    
    p = precision(y_pred, y_true, c)
    r = recall(y_pred, y_true, c)
    
    if p + r == 0:
        return 0
    else:
        return 2 * p * r / (p + r)
    
def accuracy(y_pred: pd.Series, y_true: pd.Series) -> float:

    return (y_pred == y_true).sum() / len(y_true)


#############################################

class RandomForest:
    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: int = 3,
                 min_samples_per_node: int = 1,
                 split_strategy: str = 'random',
                 subset_size_ratio: float = 0.5,
                 subset_feature_ratio: float = 0.75):

        assert 0 < subset_size_ratio <= 1, "subset_size_ratio must be between 0 and 1"
        assert 0 < subset_feature_ratio <= 1, "subset_feature_ratio must be between 0 and 1"
        
        self._trees: list[DecisionTree] = []
        self._n_estimators: int = n_estimators
        self._max_depth: int = max_depth
        self._min_samples_per_node: int = min_samples_per_node
        self._split_strategy: str = split_strategy
        self._subset_size_ratio: float = subset_size_ratio
        self._subset_feature_ratio: float = subset_feature_ratio
        
    def fit(self, X: pd.DataFrame, y: pd.Series, labels):
        for _ in range(self._n_estimators):
            indices = np.random.choice(X.shape[0], 
                                       size=int(self._subset_size_ratio * X.shape[0]), 
                                       replace=False)
            X_subset = X.iloc[indices]
            y_subset = y.iloc[indices]
            features_size = int(self._subset_size_ratio * X.shape[1])
            features = np.random.choice(X.shape[1], 
                                        size=features_size,
                                        replace=False)
            tree = DecisionTree(split_strategy=self._split_strategy,
                                max_depth=self._max_depth,
                                min_samples_per_node=self._min_samples_per_node)
            
            new_X_subset = X_subset.iloc[:, features]
            tree.fit(new_X_subset, y_subset, labels=labels)
            self._trees.append(tree)

    

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        
        predictions = []

        for tree in self._trees:
            predictions.append(tree.predict(X))
            
        return np.array([Counter(pred).most_common(1)[0][0] for pred in np.array(predictions).T])
    
    def display(self, max_trees: int = 3):
       
        for i, tree in enumerate(self._trees[:max_trees]):
            print()
            tree.display()
    
###################################

def lab_random_forest(df_train, numeric, categorical, target, df_test, labels):
    global TARGET
    TARGET = target

    X_train, y_train, X_test, y_test = prepare_data(df_train, numeric, categorical, target, df_test)

    n_estimators = 5
    max_depth = 8
    min_samples_per_node = 50
    split_strategy= 'random'
    subset_size_ratio = 0.5
    subset_feature_ratio = 1

    random_forest = RandomForest(n_estimators=n_estimators,
                                 max_depth=max_depth,
                                 min_samples_per_node=min_samples_per_node,
                                 split_strategy=split_strategy,
                                 subset_size_ratio=subset_size_ratio,
                                 subset_feature_ratio=subset_feature_ratio)
    
    random_forest.fit(X_train, y_train, labels=labels)

    # Examples of trees in the forest, remove comment to visualize 
    # random_forest.display()

    predictions = random_forest.predict(X_test)
    plot_confusion_matrix(predictions, y_test)

    return get_metrics(y_test, predictions, labels)