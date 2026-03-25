import numpy as np


def make_toy_data(n_samples: int = 200, seed: int = 42):
    '''
    Create a simple non-linear binary classification dataset.
    Label is 1 when x0 * x1 > 0, else 0, with light noise.
    '''
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, 2))
    y = ((X[:, 0] * X[:, 1]) > 0).astype(np.float64).reshape(-1, 1)

    # Add some noise so the dataset is not perfectly trivial
    X += rng.normal(scale=0.15, size=X.shape)

    return X, y


def train_val_split(X, y, val_ratio: float = 0.2, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)

    split = int(len(X) * (1 - val_ratio))
    train_idx = idx[:split]
    val_idx = idx[split:]

    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def accuracy_from_probs(probs, y_true):
    preds = (probs >= 0.5).astype(np.float64)
    return float((preds == y_true).mean())
