from dataclasses import dataclass

from src.model import SimpleMLP
from src.utils import make_toy_data, train_val_split, accuracy_from_probs


@dataclass
class TrainConfig:
    n_samples: int = 300
    hidden_dim: int = 8
    epochs: int = 300
    lr: float = 0.1
    seed: int = 42


def train_model(config: TrainConfig):
    X, y = make_toy_data(n_samples=config.n_samples, seed=config.seed)
    X_train, y_train, X_val, y_val = train_val_split(X, y, seed=config.seed)

    model = SimpleMLP(input_dim=2, hidden_dim=config.hidden_dim, seed=config.seed)

    history = []
    for epoch in range(config.epochs):
        loss, train_probs = model.train_step(X_train, y_train, lr=config.lr)
        train_acc = accuracy_from_probs(train_probs, y_train)
        val_probs = model.predict_proba(X_val)
        val_acc = accuracy_from_probs(val_probs, y_val)

        history.append(
            {
                "epoch": epoch + 1,
                "loss": float(loss),
                "train_acc": float(train_acc),
                "val_acc": float(val_acc),
            }
        )

    return model, history


def main():
    config = TrainConfig()
    _, history = train_model(config)
    last = history[-1]
    print(f"Final loss: {last['loss']:.4f}")
    print(f"Train accuracy: {last['train_acc']:.4f}")
    print(f"Validation accuracy: {last['val_acc']:.4f}")


if __name__ == "__main__":
    main()
