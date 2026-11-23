# OptiCan

OptiCan is a lightweight, framework-agnostic hyperparameter optimization library. It searches continuous, integer, categorical, Boolean, and fixed parameter spaces with adaptive elite-guided refinement.

OptiCan does not require PyTorch, TensorFlow, scikit-learn, NumPy, or another machine-learning framework. You provide an objective function that trains and validates your model. OptiCan searches the configuration values passed to that function.

The current release is `0.1.0`. The API may change before the first stable release. The search is a heuristic and does not guarantee a global optimum.

## Features

- Continuous, log-scaled, integer, categorical, Boolean, and fixed parameters.
- Adaptive range contraction guided by elite candidate density.
- Elite-parent mutation that preserves successful parameter combinations.
- Probability-based categorical refinement.
- Shared random seeds across candidate evaluations.
- Optional multi-seed stability scoring.
- Duplicate configuration avoidance and full trial history.
- Minimization and maximization objectives.
- Early stopping and trial limits.
- No required third-party dependencies.

## Install

install it with:

```bash
pip install optican
```

For local development:

```bash
python -m pip install -e ".[dev]"
pytest
```

## Quick start

An objective receives a configuration dictionary and optionally a seed. It returns one numeric validation metric. The default direction is `"min"`, which is useful for loss and error metrics.

```python
from optican import OptiCan, Float, Int, Choice, Bool


def objective(config, seed):
    validation_loss = train_and_validate(
        learning_rate=config["learning_rate"],
        batch_size=config["batch_size"],
        weight_decay=config["weight_decay"],
        optimizer=config["optimizer"],
        use_scheduler=config["use_scheduler"],
        seed=seed,
    )
    return validation_loss


search = OptiCan(
    objective=objective,
    config={
        "learning_rate": Float(1e-5, 1e-2, log=True),
        "batch_size": Choice([32, 64, 128, 256]),
        "weight_decay": Float(1e-7, 1e-2, log=True, allow_zero=True),
        "optimizer": Choice(["AdamW", "Adam", "SGD"]),
        "use_scheduler": Bool(),
    },
    rounds=3,
    ubest=False,
)

last_round_config = search.optimize()
print(last_round_config)
```

## Return behavior

`ubest=False` returns the best configuration from the latest completed round that produced a successful trial. This is useful when you want to inspect what the newest refinement round found.

`ubest=True` returns a `TuningResult`. Its `best_config`, `best_trial`, and `best_score` refer to the best result across all rounds.

The full result is available as `search.result` in both modes. It keeps the global best and the latest successful round best:

```python
print(search.result.best_config)
print(search.result.best_score)
print(search.result.last_round_best_config)
print(search.result.last_round_best_score)
print(search.result.summary())

for trial in search.result.trials:
    print(trial.config, trial.mean_score, trial.std_score, trial.error)
```

If the latest round has no successful trials, OptiCan uses the best result from the latest earlier round that had a successful trial. A fixed-only search or an empty configuration is evaluated once.

## Search-space parameters

Use helper classes for explicit search spaces:

| Type | Example | Behavior |
|---|---|---|
| Float | `Float(1e-5, 1e-2)` | Samples a continuous range |
| Log float | `Float(1e-5, 1e-2, log=True)` | Samples uniformly in log10 space |
| Zero-capable float | `Float(1e-7, 1e-2, log=True, allow_zero=True)` | Adds exact zero as a separate option |
| Integer | `Int(16, 256, step=16)` | Samples integer values on a stepped range |
| Categorical | `Choice(["Adam", "AdamW"])` | Learns category probabilities |
| Boolean | `Bool()` | Learns probabilities for `True` and `False` |
| Fixed | `Fixed("gelu")` | Keeps a value unchanged |

A list or tuple is treated as a categorical choice. Any other scalar is treated as a fixed value. An empty `config={}` is valid and evaluates the objective once with an empty configuration.

Compact dictionary definitions are also supported:

```python
config = {
    "learning_rate": {
        "type": "float",
        "low": 1e-5,
        "high": 1e-2,
        "log": True,
    },
    "depth": {"type": "int", "low": 2, "high": 12},
    "optimizer": {"type": "choice", "values": ["Adam", "AdamW"]},
    "use_dropout": {"type": "bool"},
    "activation": ["relu", "gelu", "silu"],
    "device": "cuda",
}
```

## Options

| Option | Default | Meaning |
|---|---:|---|
| `rounds` | `3` | Total search rounds, including the initial round |
| `initial_candidates` | `24` | Configurations sampled in the first round |
| `candidates_per_round` | `12` | New configurations in each later round |
| `elite_size` | `20` | Top configurations used to guide refinement |
| `seeds` | `1` | Number of shared seeds used per configuration |
| `seed` | `42` | Base seed, later seeds increase by one |
| `direction` | `"min"` | Use `"max"` for metrics such as accuracy or F1 |
| `stability_penalty` | `0.25` | Penalty for score variation across seeds |
| `mutation_rate` | `0.70` | Chance of mutating each parameter in a child |
| `exploration_mix` | `0.20` | Fraction sampled independently of elite parents |
| `categorical_exploration` | `0.15` | Keeps category sampling from collapsing too early |
| `shrink_fraction` | `0.20` | Fraction of the active numeric range contracted per round |
| `min_range_fraction` | `0.05` | Minimum range width relative to its initial width |
| `max_trials` | `None` | Optional cap on evaluated configurations |
| `patience_rounds` | `0` | Stop after this many stagnant rounds, zero disables it |
| `min_delta` | `0.0` | Minimum improvement needed to reset patience |
| `ubest` | `False` | Selects the latest-round config or returns the full global result |
| `verbose` | `True` | Prints progress while evaluating candidates |
| `catch_exceptions` | `True` | Records failed evaluations and continues |

Use `seeds=1` for faster tests. Increase it to `3` or `5` for more stable comparisons. The same seed set is used for every configuration.

OptiCan evaluates candidates sequentially by default. This avoids global random-seed interference and unexpected GPU memory use across different frameworks. Reduce candidate counts, use `max_trials`, or add early stopping inside the objective to reduce evaluation time.

## Metric formulas

A separate `metric="t/a"` option is not needed. The objective function already provides a flexible way to calculate any metric. Compute the values your formula needs and return the result.

```python
def objective(config, seed):
    predictions = train_and_predict(config, seed)
    t = count_true_predictions(predictions)
    a = count_all_predictions(predictions)
    return t / max(a, 1)
```

The objective can calculate a single metric, combine several metrics, or call an existing scoring function. OptiCan only needs the final numeric result. Set `direction="max"` when higher values are better.

## PyTorch example

OptiCan does not create or manage models itself. Put model creation, training, early stopping, and validation inside the objective. Replace `Model` and the data variables with your own objects.

```python
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from optican import OptiCan, Float, Choice, Bool


def objective(config, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model().to(device)
    optimizer_class = getattr(torch.optim, config["optimizer"])
    optimizer = optimizer_class(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    loss_fn = nn.CrossEntropyLoss()
    scheduler = None
    if config["use_scheduler"]:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=config["batch_size"],
        shuffle=True,
    )

    best_val_loss = float("inf")
    for _ in range(20):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(
                model(X_valid.to(device)),
                y_valid.to(device),
            ).item()
        best_val_loss = min(best_val_loss, val_loss)
        if scheduler is not None:
            scheduler.step()

    return best_val_loss


search = OptiCan(
    objective,
    config={
        "learning_rate": Float(1e-5, 3e-3, log=True),
        "weight_decay": Float(1e-7, 1e-2, log=True, allow_zero=True),
        "batch_size": Choice([32, 64, 128, 256]),
        "optimizer": Choice(["AdamW", "Adam"]),
        "use_scheduler": Bool(),
    },
    rounds=3,
    seeds=3,
    ubest=True,
)

result = search.optimize()
print(result.best_config)
print(result.best_score)
```

## Scikit-learn example

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from optican import OptiCan, Int, Choice


def objective(config, seed):
    estimator = RandomForestClassifier(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        criterion=config["criterion"],
        random_state=seed,
        n_jobs=-1,
    )
    return cross_val_score(
        estimator, X, y, cv=5, scoring="accuracy", n_jobs=-1
    ).mean()


search = OptiCan(
    objective,
    config={
        "n_estimators": Int(100, 600, step=50),
        "max_depth": Choice([None, 5, 10, 20, 30]),
        "criterion": Choice(["gini", "entropy", "log_loss"]),
    },
    direction="max",
    rounds=3,
)

best_config = search.optimize()
```

## How the algorithm works

1. Initial configurations are sampled from the parameter distributions.
2. Every configuration is evaluated with the same seed set.
3. Successful trials are ranked by their mean metric and optional stability penalty.
4. Numeric ranges contract toward elite regions, with more contraction on the side with fewer elite samples.
5. Categorical probabilities increase for values that appear more often among elite trials.
6. New configurations mutate successful parent configurations, preserving useful parameter combinations.
7. Every trial is kept, so the global best can be selected across all rounds.

This is an adaptive heuristic. It can miss a global optimum, especially with narrow ranges, noisy objectives, strong parameter interactions, or too few evaluations. Use a suitable evaluation budget and validation split.

## Reproducibility and objective requirements

- Return one finite numeric validation metric.
- Do not use the test set to choose hyperparameters. Reserve it for final evaluation.
- Use the supplied seed for model initialization, data shuffling, and randomized preprocessing.
- OptiCan seeds Python and seeds NumPy, PyTorch, and TensorFlow only when they are already imported.
- Some GPU operations may remain nondeterministic unless the framework is configured for deterministic execution.
- Add early stopping inside the objective when the training loop supports it.

## Development

```bash
python -m pip install -e ".[dev]"
pytest
ruff check src tests examples
python -m build
python -m twine check dist/*
```

## License

MIT. See [LICENSE](LICENSE).
