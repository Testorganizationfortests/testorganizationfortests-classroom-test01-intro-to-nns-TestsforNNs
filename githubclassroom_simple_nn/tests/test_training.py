import numpy as np

from src.model import SimpleMLP
from src.utils import make_toy_data, accuracy_from_probs


def test_one_step_reduces_loss():
    X, y = make_toy_data(n_samples=80, seed=1)
    model = SimpleMLP(input_dim=2, hidden_dim=8, seed=1)

    initial_probs = model.forward(X)
    initial_loss = model.compute_loss(initial_probs, y)

    for _ in range(20):
        model.train_step(X, y, lr=0.1)

    final_probs = model.forward(X)
    final_loss = model.compute_loss(final_probs, y)

    assert final_loss < initial_loss


def test_model_learns_reasonable_accuracy():
    X, y = make_toy_data(n_samples=120, seed=2)
    model = SimpleMLP(input_dim=2, hidden_dim=8, seed=2)

    for _ in range(250):
        model.train_step(X, y, lr=0.1)

    probs = model.predict_proba(X)
    acc = accuracy_from_probs(probs, y)

    assert acc >= 0.75
