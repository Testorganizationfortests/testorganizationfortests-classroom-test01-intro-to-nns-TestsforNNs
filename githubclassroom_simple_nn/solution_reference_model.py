# Instructor reference solution.
# Do not give this file to students in the actual assignment repo.

import numpy as np


class SimpleMLP:
    def __init__(self, input_dim: int = 2, hidden_dim: int = 8, seed: int = 42):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.rng = np.random.default_rng(seed)

        self.cache = {}
        self.grads = {}
        self._init_params()

    def _init_params(self):
        self.W1 = self.rng.normal(0.0, 0.2, size=(self.input_dim, self.hidden_dim))
        self.b1 = np.zeros((1, self.hidden_dim), dtype=np.float64)
        self.W2 = self.rng.normal(0.0, 0.2, size=(self.hidden_dim, 1))
        self.b2 = np.zeros((1, 1), dtype=np.float64)

    @staticmethod
    def sigmoid(z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def tanh(z):
        return np.tanh(z)

    @staticmethod
    def tanh_derivative(a):
        return 1.0 - a ** 2

    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = self.tanh(z1)
        z2 = a1 @ self.W2 + self.b2
        y_hat = self.sigmoid(z2)

        self.cache = {"X": X, "z1": z1, "a1": a1, "z2": z2, "y_hat": y_hat}
        return y_hat

    def compute_loss(self, y_hat, y):
        eps = 1e-8
        y_hat = np.clip(y_hat, eps, 1.0 - eps)
        loss = -(y * np.log(y_hat) + (1.0 - y) * np.log(1.0 - y_hat)).mean()
        return float(loss)

    def backward(self, X, y):
        m = X.shape[0]
        a1 = self.cache["a1"]
        y_hat = self.cache["y_hat"]

        dz2 = (y_hat - y) / m
        dW2 = a1.T @ dz2
        db2 = dz2.sum(axis=0, keepdims=True)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * self.tanh_derivative(a1)
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0, keepdims=True)

        self.grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    def update_params(self, lr: float = 0.1):
        self.W1 -= lr * self.grads["dW1"]
        self.b1 -= lr * self.grads["db1"]
        self.W2 -= lr * self.grads["dW2"]
        self.b2 -= lr * self.grads["db2"]

    def train_step(self, X, y, lr: float = 0.1):
        y_hat = self.forward(X)
        loss = self.compute_loss(y_hat, y)
        self.backward(X, y)
        self.update_params(lr=lr)
        return loss, y_hat

    def predict_proba(self, X):
        return self.forward(X)

    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs >= 0.5).astype(np.float64)
