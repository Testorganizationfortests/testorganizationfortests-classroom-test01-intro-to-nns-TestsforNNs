import numpy as np

from src.model import SimpleMLP


def test_parameter_shapes():
    model = SimpleMLP(input_dim=2, hidden_dim=5, seed=0)
    assert model.W1.shape == (2, 5)
    assert model.b1.shape == (1, 5)
    assert model.W2.shape == (5, 1)
    assert model.b2.shape == (1, 1)


def test_forward_output_shape():
    model = SimpleMLP(input_dim=2, hidden_dim=4, seed=0)
    X = np.array([[0.1, -0.2], [1.0, 0.5], [-0.3, 0.2]], dtype=np.float64)
    y_hat = model.forward(X)
    assert y_hat.shape == (3, 1)
    assert np.all(y_hat >= 0.0) and np.all(y_hat <= 1.0)
