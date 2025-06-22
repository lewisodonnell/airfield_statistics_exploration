
"""
Simple 3-layer MLP (ReLU hidden layers, linear output) trained with mini-batch
SGD and the compound loss.
"""

from __future__ import annotations
from typing import List
import numpy as np
from tqdm import tqdm

from models.losses import compound_loss, grad_compound_loss



def dense(X, W, b):
    return X @ W.T + b


def relu(x):
    return np.maximum(x, 0.0)


def grad_relu(x):
    g = np.zeros_like(x)
    g[x > 0] = 1.0
    return g


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def grad_sigmoid(x):
    s = sigmoid(x)
    return s * (1.0 - s)

_activation = {
    "relu":     (relu, grad_relu),
    "identity": (lambda x: x, lambda x: np.ones_like(x)),
    "sigmoid":  (sigmoid, grad_sigmoid),   # ← added
}




class MLP:
    def __init__(self, seed: int = 2):
        self.layers: List[dict] = []
        self.rng = np.random.default_rng(seed)
        self.loss_history_: List[float] = []

    
    def add_layer(self, in_dim: int, out_dim: int, act: str = "identity"):
        W = self.rng.normal(size=(out_dim, in_dim)) * np.sqrt(2.0 / (in_dim + out_dim))
        b = np.zeros(out_dim)
        self.layers.append({"W": W, "b": b, "act": act})

    
    def _forward(self, X):
      
        if X.ndim == 1:
            X = X.reshape(1, -1)

        h = X
        graph = []

        for lyr in self.layers:
            a = dense(h, lyr["W"], lyr["b"])
            h_next = _activation[lyr["act"]][0](a)
            graph.append({"h_in": h, "a": a, "h": h_next})
            h = h_next                         

        return h, graph                        


    
    def _backprop(self, graph, delta_out):
        """
        graph      : list produced by _forward (len == #layers)
        delta_out  : ∂Loss / ∂y_hat              (batch × 2)

        Returns
        -------
        grads in the *same order* as self.layers, each dict has 'W', 'b'
        """
        grads = []
        delta = delta_out                                    

        
        for idx in range(len(self.layers) - 1, -1, -1):
            lyr   = self.layers[idx]
            g_cur = graph[idx]               

            
            grad_W = delta.T @ g_cur["h_in"] / g_cur["h_in"].shape[0]
            grad_b = delta.mean(axis=0)
            grads.append({"W": grad_W, "b": grad_b})

            
            if idx == 0:
                break

            
            dh = delta @ lyr["W"]                           
            prev_a   = graph[idx - 1]["a"]
            prev_act = self.layers[idx - 1]["act"]
            delta = dh * _activation[prev_act][1](prev_a)    

        return list(reversed(grads))

    
    def fit(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        lam: float = 0.0,
        lr: float = 5e-5,
        epochs: int = 200,
        batch: int = 20,
        seed: int = 0,
    ):
        rng = np.random.default_rng(seed)
        n_batches = len(y_train) // batch

        for _ in tqdm(range(epochs), leave=False):
            order = rng.permutation(len(y_train))
            X_shuf, y_shuf = X_train[order], y_train[order]

            for k in range(n_batches):
                xb = X_shuf[k * batch : (k + 1) * batch]
                yb = y_shuf[k * batch : (k + 1) * batch]

                y_hat, graph = self._forward(xb)
                delta_out = grad_compound_loss(yb, y_hat, lam=lam)
                grads = self._backprop(graph, delta_out)

               
                for lyr, g in zip(self.layers, grads):
                    lyr["W"] -= lr * g["W"]
                    lyr["b"] -= lr * g["b"]

           
            y_hat_train, _ = self._forward(X_train)
            self.loss_history_.append(float(compound_loss(y_train, y_hat_train, lam=lam)))
        return self

    
    def predict(self, X):
        y_hat, _ = self._forward(X)
        return y_hat
