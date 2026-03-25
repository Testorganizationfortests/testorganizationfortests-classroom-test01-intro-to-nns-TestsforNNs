import numpy as np

from src.model import SimpleMLP


def test_loss_is_scalar_and_positive():
    model = SimpleMLP(input_dim=2, hidden_dim=3, seed=0)
    X = np.array([[0.5, -0.2], [0.1, 0.7]], dtype=np.float64)
    y = np.array([[1.0], [0.0]], dtype=np.float64)
    y_hat = model.forward(X)
    loss = model.compute_loss(y_hat, y)

    assert np.isscalar(loss) or getattr(loss, "shape", ()) == ()
    assert loss >= 0.0


def test_forward_known_parameters():
    model = SimpleMLP(input_dim=2, hidden_dim=2, seed=0)

    model.W1 = np.array([[1.0, -1.0], [0.5, 0.5]])
    model.b1 = np.array([[0.0, 0.0]])
    model.W2 = np.array([[1.0], [-1.0]])
    model.b2 = np.array([[0.0]])

    X = np.array([[1.0, 2.0]], dtype=np.float64)
    y_hat = model.forward(X)

    z1 = X @ model.W1 + model.b1
    a1 = np.tanh(z1)
    z2 = a1 @ model.W2 + model.b2
    expected = 1.0 / (1.0 + np.exp(-z2))

    assert np.allclose(y_hat, expected, atol=1e-8)
