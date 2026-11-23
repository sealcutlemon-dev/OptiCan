# Contributing

Contributions, bug reports, and reproducible benchmark results are welcome.

## Before opening a pull request

- Create a virtual environment.
- Install development dependencies with `python -m pip install -e ".[dev]"`.
- Add or update tests for behavior changes.
- Run `pytest` and `ruff check src tests examples`.
- Include the search space, objective, Python version, and random seed when reporting performance changes.

Do not claim that the search is globally optimal. Use the same evaluation budget, seeds, validation splits, and stopping rules when comparing with other search methods. Do not use a held-out test set for tuning.
