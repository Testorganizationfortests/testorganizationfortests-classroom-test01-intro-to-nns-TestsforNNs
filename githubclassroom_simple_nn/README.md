# Simple Neural Network Assignment (NumPy)

This is a starter repository for a GitHub Classroom assignment.

Students implement a small 2-layer neural network for binary classification.

## Learning goals

- understand forward propagation
- compute binary cross-entropy loss
- implement backpropagation for a tiny MLP
- train on a toy dataset
- interpret training behavior

## What students should edit

Students should mainly edit:

- `src/model.py`

They may inspect but usually should not need to modify:

- `src/train.py`
- `src/utils.py`

## Tasks

Implement the TODOs in `src/model.py`:

- parameter initialization
- forward pass
- binary cross-entropy loss
- backward pass
- parameter update
- one training step

## Install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run tests

```bash
pytest -q
```

## Run training

```bash
python -m src.train
```

## Grading idea

- shapes and API
- forward pass correctness
- loss correctness
- one training step decreases loss
- training reaches a minimum accuracy

## Notes for instructors

- keep datasets tiny and deterministic
- use hidden tests for edge cases, not gotchas
- avoid exact float equality
