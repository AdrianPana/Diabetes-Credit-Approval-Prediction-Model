import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from typing import List
import numpy as np
import matplotlib.pyplot as plt

from utils import prepare_data, plot_confusion_matrix, get_metrics

def scikit_mlp(df_train, numeric, categorical, target, df_test, labels):
        
    X_train, y_train, X_test, y_test = prepare_data(df_train, numeric, categorical, target, df_test)

    hidden_layer_sizes = [100, 100]
    activation = 'relu'
    solver = 'lbfgs'
    learning_rate = 'invscaling'
    alpha = 0.001
    max_iter = 300


    mlp = MLPClassifier(random_state=1,
                        max_iter=max_iter,
                        hidden_layer_sizes=hidden_layer_sizes,
                        activation=activation,
                        solver=solver,
                        learning_rate=learning_rate,
                        alpha=alpha).fit(X_train, y_train)
    
    predictions = mlp.predict(X_test)

    plot_confusion_matrix(predictions, y_test)

    return get_metrics(y_test, predictions, labels)

##############################################

class Layer:

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError
        
    def backward(self, x: np.ndarray, dy: np.ndarray) -> np.ndarray:
        raise NotImplementedError
        
    def update(self, *args, **kwargs):
        pass

class FeedForwardNetwork:
    
    def __init__(self, layers: List[Layer]):
        self.layers = layers
        
    def forward(self, x: np.ndarray, train: bool = True) -> np.ndarray:
        self._inputs = []
        for layer in self.layers:
            if train:
                self._inputs.append(x)
            x = layer.forward(x)
        return x
    
    def backward(self, dy: np.ndarray) -> np.ndarray:
        for i in range(len(self.layers) - 1, -1, -1):
            x = self._inputs[i]
            dy = self.layers[i].backward(x, dy)
    
        del self._inputs
    
    def update(self, *args, **kwargs):
        for layer in self.layers:
            layer.update(*args, **kwargs)

class Linear(Layer):
    
    def __init__(self, insize: int, outsize: int) -> None:
        bound = np.sqrt(6. / insize)
        
        self.weight = np.random.uniform(-bound, bound, (insize, outsize)).astype(np.float64)
        self.bias = np.zeros((outsize,), dtype=np.float64)

        self.dweight = np.zeros_like(self.weight, dtype=np.float64)
        self.dbias = np.zeros_like(self.bias, dtype=np.float64)
   
    def forward(self, x: np.ndarray) -> np.ndarray:        
        return np.dot(x, self.weight) + self.bias
    
    def backward(self, x: np.ndarray, dy: np.ndarray) -> np.ndarray:
        self.dweight = np.dot(x.T, dy)
        self.dbias = np.sum(dy, axis=0)
        return np.dot(dy, self.weight.T)
    
    def update(self, mode='SGD', lr=0.001, mu=0.9):
        if mode == 'SGD':
            self.dweight = self.dweight.astype(np.float64)
            self.dbias = self.dbias.astype(np.float64)
            self.weight -= lr * self.dweight
            self.bias -= lr * self.dbias
        else:
            raise ValueError('mode should be SGD, not ' + str(mode))
        
class ReLU(Layer):
    
    def __init__(self) -> None:
        pass
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0)
    
    def backward(self, x: np.ndarray, dy: np.ndarray) -> np.ndarray:
        gradient = x > 0
        return dy * gradient.astype(int)
    
class CrossEntropy:
    
    def __init__(self):
        pass
    
    def softmax(self, x):
        arr_x = np.array(x, dtype=np.float64)
        exps = np.exp(arr_x - np.max(arr_x, axis=1, keepdims=True))  # for numerical stability
        return exps / np.sum(exps, axis=1, keepdims=True)

    def forward(self, y: np.ndarray, t: np.ndarray) -> float:
        pk = self.softmax(y)
        pt = pk[np.arange(len(t)), t]
        l = -np.log(pt)
        res = np.sum(l) / t.size
        return res
    
    def backward(self, y: np.ndarray, t: np.ndarray) -> np.ndarray:
        pk = self.softmax(y)
        pk[np.arange(len(t)), t] -= 1
        return pk / t.size
        
def accuracy(y: np.ndarray, t: np.ndarray) -> float:
    predictions = np.argmax(y, axis = 1)
    correct = np.sum(predictions == t)
    return correct / t.size

def lab_mlp(df_train, numeric, categorical, target, df_test, labels):
    BATCH_SIZE = 1000
    HIDDEN_NODES = 100
    EPOCHS_NO = 10

    optimize_args = {'mode': 'SGD', 'lr': .05}

    cost_function = CrossEntropy()

    X_train, y_train, X_test, y_test = prepare_data(df_train, numeric, categorical, target, df_test)

    INPUT_NODES = X_train.shape[1]
    OUTPUT_NODES = len(labels)

    predictions = None


    net = FeedForwardNetwork([
                            Linear(INPUT_NODES, HIDDEN_NODES),
                            Linear(HIDDEN_NODES, OUTPUT_NODES),
                            ReLU(),
                            ])
    
    for epoch in range(EPOCHS_NO):
        for b_no, idx in enumerate(range(0, len(X_train), BATCH_SIZE)):
            x = X_train.iloc[idx:idx + BATCH_SIZE].values
            t = y_train[idx:idx + BATCH_SIZE]
            
            y = net.forward(x)
            loss = cost_function.forward(y, t)
            grad_err = cost_function.backward(y, t)
            grad = net.backward(grad_err)
            
            net.update(**optimize_args)

        y = net.forward(X_test.values, train=False)
        predictions = np.argmax(y, axis = 1)

    plot_confusion_matrix(predictions, y_test)

    return get_metrics(y_test, predictions, labels)