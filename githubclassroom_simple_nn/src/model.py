import numpy as np


class SimpleMLP:
    '''
    A tiny 2-layer neural network for binary classification.

    Architecture:
        input -> hidden(tanh) -> output(sigmoid)
    '''

    def __init__(self, input_dim: int = 2, hidden_dim: int = 8, seed: int = 42):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.rng = np.random.default_rng(seed)

        # TODO: initialize parameters
        # Suggested shapes:
        # W1: (input_dim, hidden_dim)
        # b1: (1, hidden_dim)
        # W2: (hidden_dim, 1)
        # b2: (1, 1)
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None

        # Caches for backprop
        self.cache = {}
        self.grads = {}

        self._init_params()

    def _init_params(self):
        '''
        Initialize weights with small random values and biases with zeros.
        '''
        # TODO: implement
        raise NotImplementedError("Implement parameter initialization.")

    @staticmethod
    def sigmoid(z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def tanh(z):
        return np.tanh(z)

    @staticmethod
    def tanh_derivative(a):
        '''
        Derivative of tanh when given the already-activated value a = tanh(z).
        '''
        return 1.0 - a ** 2

    def forward(self, X):
        '''
        Returns predicted probabilities of shape (batch_size, 1).
        '''
        # TODO: implement hidden layer and output layer
        # Save intermediate tensors in self.cache for backprop.
        raise NotImplementedError("Implement the forward pass.")

    def compute_loss(self, y_hat, y):
        '''
        Binary cross-entropy loss.
        '''
        # TODO: implement BCE
        raise NotImplementedError("Implement the loss function.")

    def backward(self, X, y):
        '''
        Compute gradients and store them in self.grads.
        Assumes forward() has already been called.
        '''
        # TODO: implement backpropagation
        raise NotImplementedError("Implement backward propagation.")

    def update_params(self, lr: float = 0.1):
        '''
        Gradient descent parameter update.
        '''
        # TODO: apply gradients
        raise NotImplementedError("Implement parameter updates.")

    def train_step(self, X, y, lr: float = 0.1):
        '''
        One training step: forward -> loss -> backward -> update.
        Returns (loss, predictions).
        '''
        # TODO: implement one full training step
        raise NotImplementedError("Implement the train_step method.")

    def predict_proba(self, X):
        return self.forward(X)

    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs >= 0.5).astype(np.float64)
