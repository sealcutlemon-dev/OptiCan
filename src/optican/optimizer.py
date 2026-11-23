from __future__ import annotations

import inspect
import math
import random
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from .space import Choice, Fixed, Float, Int, normalize_parameter


@dataclass
class TrialResult:
    """One evaluated configuration."""

    config: Dict[str, Any]
    scores: Tuple[float, ...]
    mean_score: float
    std_score: float
    selection_score: float
    seeds: Tuple[int, ...]
    error: Optional[str] = None
    duration: float = 0.0

    @property
    def successful(self) -> bool:
        return self.error is None and math.isfinite(self.selection_score)


@dataclass
class TuningResult:
    """Complete result of an optimization run."""

    best_config: Dict[str, Any]
    best_score: float
    best_selection_score: float
    best_trial: TrialResult
    last_round_best_config: Dict[str, Any]
    last_round_best_score: float
    last_round_best_trial: TrialResult
    trials: List[TrialResult]
    rounds_completed: int
    elapsed_seconds: float

    @property
    def successful_trials(self) -> List[TrialResult]:
        return [trial for trial in self.trials if trial.successful]

    @property
    def failed_trials(self) -> List[TrialResult]:
        return [trial for trial in self.trials if not trial.successful]

    def summary(self) -> Dict[str, Any]:
        return {
            "best_config": dict(self.best_config),
            "best_score": self.best_score,
            "best_selection_score": self.best_selection_score,
            "last_round_best_config": dict(self.last_round_best_config),
            "last_round_best_score": self.last_round_best_score,
            "trials": len(self.trials),
            "successful_trials": len(self.successful_trials),
            "failed_trials": len(self.failed_trials),
            "rounds_completed": self.rounds_completed,
            "elapsed_seconds": self.elapsed_seconds,
        }


class OptiCan:
    """Search mixed hyperparameter spaces using adaptive elite sampling.

    Parameters:
        objective: A callable returning a scalar validation metric. It may
            accept ``objective(config)`` or ``objective(config, seed)``. The
            optimizer minimizes the metric by default.
        config: Search-space dictionary. This is an alias for ``search_space``.
            Values may be ``Float``, ``Int``, ``Choice``, ``Bool``, lists of
            choices, specification dictionaries, or fixed scalar values.
        search_space: Alternative, more explicit name for ``config``.
        rounds: Total number of rounds, including the initial sampling round.
        ubest: When False, ``optimize()`` returns the best config from the latest
            successful round. When True, it returns the full result with the
            best trial selected across all rounds.
        initial_candidates: Number of candidates in round one.
        candidates_per_round: Number of candidates in each later round.
        elite_size: Number of top candidates used to guide refinement.
        seeds: Number of deterministic objective evaluations per config.
        seed: Base seed. Every candidate is evaluated with the same seed set.
        direction: ``"min"`` or ``"max"``.
        stability_penalty: Penalty applied to standard deviation across seeds.
        mutation_rate: Probability that a child mutates each parameter.
        exploration_mix: Fraction of new candidates sampled independently
            rather than from elite parents.
        categorical_exploration: Uniform exploration mixed into category rates.
        shrink_fraction: Fraction of the active numeric range contracted each
            refinement round.
        min_range_fraction: Numeric range floor as a fraction of its initial
            width, preventing premature collapse.
        max_trials: Optional upper bound on evaluated configurations.
        patience_rounds: Stop after this many rounds without improvement;
            zero disables this early stop.
        min_delta: Minimum selection-score improvement to reset patience.
        catch_exceptions: Record failed trials and continue when True.
        verbose: Print a compact progress log.
        on_trial: Optional callback invoked after each candidate evaluation.
    """

    def __init__(
        self,
        objective: Callable[..., float],
        config: Optional[Mapping[str, Any]] = None,
        *,
        search_space: Optional[Mapping[str, Any]] = None,
        rounds: int = 3,
        ubest: bool = False,
        initial_candidates: int = 24,
        candidates_per_round: int = 12,
        elite_size: int = 20,
        seeds: int = 1,
        seed: int = 42,
        direction: str = "min",
        stability_penalty: float = 0.25,
        mutation_rate: float = 0.70,
        exploration_mix: float = 0.20,
        categorical_exploration: float = 0.15,
        shrink_fraction: float = 0.20,
        min_range_fraction: float = 0.05,
        max_trials: Optional[int] = None,
        patience_rounds: int = 0,
        min_delta: float = 0.0,
        catch_exceptions: bool = True,
        verbose: bool = True,
        on_trial: Optional[Callable[[TrialResult], None]] = None,
    ) -> None:
        if not callable(objective):
            raise TypeError("objective must be callable.")
        if config is not None and search_space is not None:
            raise ValueError("Pass either config= or search_space=, not both.")
        raw_space = search_space if search_space is not None else config
        if raw_space is None:
            raise ValueError("Provide config={} or search_space={}.")
        if not isinstance(raw_space, Mapping):
            raise TypeError("config/search_space must be a mapping.")
        if not all(isinstance(name, str) and name for name in raw_space):
            raise ValueError("All parameter names must be non-empty strings.")

        integer_options = {
            "rounds": rounds,
            "initial_candidates": initial_candidates,
            "candidates_per_round": candidates_per_round,
            "elite_size": elite_size,
            "seeds": seeds,
            "seed": seed,
            "patience_rounds": patience_rounds,
        }
        for option_name, option_value in integer_options.items():
            if isinstance(option_value, bool) or not isinstance(option_value, Integral):
                raise TypeError(f"{option_name} must be an integer.")

        rounds = int(rounds)
        initial_candidates = int(initial_candidates)
        candidates_per_round = int(candidates_per_round)
        elite_size = int(elite_size)
        seeds = int(seeds)
        seed = int(seed)
        patience_rounds = int(patience_rounds)

        if rounds < 1:
            raise ValueError("rounds must be >= 1.")
        if initial_candidates < 1 or candidates_per_round < 1:
            raise ValueError("Candidate counts must be >= 1.")
        if elite_size < 1:
            raise ValueError("elite_size must be >= 1.")
        if seeds < 1:
            raise ValueError("seeds must be >= 1.")
        if seed < 0:
            raise ValueError("seed must be >= 0.")
        if direction not in {"min", "max"}:
            raise ValueError("direction must be 'min' or 'max'.")
        if not isinstance(ubest, bool):
            raise TypeError("ubest must be a boolean.")

        numeric_options = {
            "stability_penalty": stability_penalty,
            "mutation_rate": mutation_rate,
            "exploration_mix": exploration_mix,
            "categorical_exploration": categorical_exploration,
            "shrink_fraction": shrink_fraction,
            "min_range_fraction": min_range_fraction,
            "min_delta": min_delta,
        }
        for name, value in numeric_options.items():
            if isinstance(value, bool):
                raise TypeError(f"{name} must be numeric.")
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise TypeError(f"{name} must be numeric.") from exc
            if not math.isfinite(number):
                raise ValueError(f"{name} must be finite.")
            numeric_options[name] = number

        stability_penalty = numeric_options["stability_penalty"]
        mutation_rate = numeric_options["mutation_rate"]
        exploration_mix = numeric_options["exploration_mix"]
        categorical_exploration = numeric_options["categorical_exploration"]
        shrink_fraction = numeric_options["shrink_fraction"]
        min_range_fraction = numeric_options["min_range_fraction"]
        min_delta = numeric_options["min_delta"]

        if stability_penalty < 0:
            raise ValueError("stability_penalty must be non-negative.")
        for name, value in (
            ("mutation_rate", mutation_rate),
            ("exploration_mix", exploration_mix),
            ("categorical_exploration", categorical_exploration),
            ("shrink_fraction", shrink_fraction),
            ("min_range_fraction", min_range_fraction),
        ):
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1.")
        if max_trials is not None:
            if isinstance(max_trials, bool) or not isinstance(max_trials, Integral):
                raise TypeError("max_trials must be an integer.")
            max_trials = int(max_trials)
            if max_trials < 1:
                raise ValueError("max_trials must be >= 1 when supplied.")
        if patience_rounds < 0:
            raise ValueError("patience_rounds cannot be negative.")
        if min_delta < 0:
            raise ValueError("min_delta must be non-negative.")
        if not isinstance(catch_exceptions, bool) or not isinstance(verbose, bool):
            raise TypeError("catch_exceptions and verbose must be booleans.")
        if on_trial is not None and not callable(on_trial):
            raise TypeError("on_trial must be callable when supplied.")

        self.objective = objective
        self.space = {name: normalize_parameter(value) for name, value in raw_space.items()}
        assert all(isinstance(spec, (Float, Int, Choice, Fixed)) for spec in self.space.values()), "OptiCan normalized an unsupported parameter. Check normalize_parameter."
        self.rounds = rounds
        self.ubest = ubest
        self.initial_candidates = initial_candidates
        self.candidates_per_round = candidates_per_round
        self.elite_size = elite_size
        self.seeds = tuple(seed + offset for offset in range(seeds))
        self.seed = seed
        self.direction = direction
        self.stability_penalty = stability_penalty
        self.mutation_rate = mutation_rate
        self.exploration_mix = exploration_mix
        self.categorical_exploration = categorical_exploration
        self.shrink_fraction = shrink_fraction
        self.min_range_fraction = min_range_fraction
        self.max_trials = max_trials
        self.patience_rounds = patience_rounds
        self.min_delta = min_delta
        self.catch_exceptions = catch_exceptions
        self.verbose = verbose
        self.on_trial = on_trial

        self._rng = random.Random(seed)
        self._states = self._initial_states()
        expected_states = {
            name for name, spec in self.space.items()
            if isinstance(spec, (Float, Int, Choice))
        }
        assert set(self._states) == expected_states, "OptiCan created an invalid search state. Check parameter initialization."
        self._objective_mode = self._inspect_objective(objective)
        self.result: Optional[TuningResult] = None

    @property
    def best_config(self) -> Optional[Dict[str, Any]]:
        return None if self.result is None else dict(self.result.best_config)

    @property
    def last_round_best_config(self) -> Optional[Dict[str, Any]]:
        if self.result is None:
            return None
        return dict(self.result.last_round_best_config)

    def optimize(self) -> Union[Dict[str, Any], TuningResult]:
        """Run the search and return the latest-round config or full result.

        ``ubest=False`` returns the best config from the latest successful
        round. ``ubest=True`` returns the full result, whose best trial is
        selected across all completed rounds.
        """
        start_time = time.perf_counter()
        self._rng = random.Random(self.seed)
        self._states = self._initial_states()
        self.result = None
        cache: Dict[Any, TrialResult] = {}
        rounds_completed = 0
        previous_best: Optional[float] = None
        stagnant_rounds = 0
        last_round_best: Optional[TrialResult] = None

        has_tunable_parameters = any(self._is_tunable(spec) for spec in self.space.values())
        total_rounds = self.rounds if has_tunable_parameters else 1

        for round_number in range(total_rounds):
            if self.max_trials is not None:
                remaining = self.max_trials - len(cache)
                if remaining <= 0:
                    break
            else:
                remaining = None

            if round_number == 0:
                count = self.initial_candidates if has_tunable_parameters else 1
                elites: List[TrialResult] = []
            else:
                elites = self._select_elites(list(cache.values()))
                if not elites:
                    break
                self._update_states(elites)
                count = self.candidates_per_round

            if remaining is not None:
                count = min(count, remaining)

            candidates = self._generate_candidates(count, cache, elites)
            if not candidates:
                break

            rounds_completed += 1
            if self.verbose:
                print(
                    f"Round {rounds_completed}/{self.rounds}: "
                    f"evaluating {len(candidates)} configuration(s)"
                )

            round_trials: List[TrialResult] = []
            for index, candidate in enumerate(candidates, 1):
                assert set(candidate) == set(self.space), "OptiCan generated an incomplete configuration. Check the parameter sampler."
                key = self._config_key(candidate)
                trial = self._evaluate(candidate)
                cache[key] = trial
                round_trials.append(trial)

                if self.on_trial is not None:
                    self.on_trial(trial)

                if self.verbose:
                    if trial.successful:
                        print(
                            f"  {index}/{len(candidates)} "
                            f"mean={trial.mean_score:.6g} "
                            f"std={trial.std_score:.3g} "
                            f"selected={trial.selection_score:.6g}"
                        )
                    else:
                        print(f"  {index}/{len(candidates)} failed: {trial.error}")

            completed_round_elites = self._select_elites(round_trials)
            if completed_round_elites:
                last_round_best = completed_round_elites[0]

            round_elites = self._select_elites(list(cache.values()))
            if not round_elites:
                continue

            current_best = round_elites[0].selection_score
            if self.verbose:
                print(
                    f"  Best mean score so far: "
                    f"{round_elites[0].mean_score:.6g}\n"
                )

            if previous_best is None or self._improved(current_best, previous_best):
                stagnant_rounds = 0
            else:
                stagnant_rounds += 1
            previous_best = current_best

            if self.patience_rounds and stagnant_rounds >= self.patience_rounds:
                if self.verbose:
                    print("Stopping early: no meaningful improvement.")
                break

        successful = self._select_elites(list(cache.values()))
        if not successful:
            errors = [trial.error for trial in cache.values() if trial.error]
            detail = errors[0] if errors else "No candidate could be evaluated."
            raise RuntimeError(f"All candidate evaluations failed. First error: {detail}")

        best = successful[0]
        if last_round_best is None:
            last_round_best = best
        self.result = TuningResult(
            best_config=dict(best.config),
            best_score=best.mean_score,
            best_selection_score=best.selection_score,
            best_trial=best,
            last_round_best_config=dict(last_round_best.config),
            last_round_best_score=last_round_best.mean_score,
            last_round_best_trial=last_round_best,
            trials=list(cache.values()),
            rounds_completed=rounds_completed,
            elapsed_seconds=time.perf_counter() - start_time,
        )

        if self.verbose:
            print("Search complete.")
            print(f"Best score: {self.result.best_score:.6g}")
            print(f"Best config: {self.result.best_config}")

        return self.result if self.ubest else dict(self.result.last_round_best_config)

    @staticmethod
    def _is_tunable(spec: Any) -> bool:
        if isinstance(spec, Float):
            return True
        if isinstance(spec, Int):
            return (spec.high - spec.low) // spec.step > 0
        if isinstance(spec, Choice):
            return len(spec.values) > 1
        return False

    def _initial_states(self) -> Dict[str, Dict[str, Any]]:
        states: Dict[str, Dict[str, Any]] = {}
        for name, spec in self.space.items():
            if isinstance(spec, Float):
                low = math.log10(spec.low) if spec.log else float(spec.low)
                high = math.log10(spec.high) if spec.log else float(spec.high)
                states[name] = {
                    "low": low,
                    "high": high,
                    "initial_low": low,
                    "initial_high": high,
                    "zero_probability": spec.zero_probability if spec.allow_zero else 0.0,
                }
            elif isinstance(spec, Int):
                states[name] = {
                    "low": float(spec.low),
                    "high": float(spec.high),
                    "initial_low": float(spec.low),
                    "initial_high": float(spec.high),
                }
            elif isinstance(spec, Choice):
                states[name] = {
                    "probabilities": [1.0 / len(spec.values)] * len(spec.values)
                }
        return states

    @staticmethod
    def _inspect_objective(objective: Callable[..., float]) -> str:
        try:
            signature = inspect.signature(objective)
        except (TypeError, ValueError):
            return "two"

        parameters = signature.parameters
        if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
            return "keyword_seed"

        seed_parameter = parameters.get("seed")
        if seed_parameter is not None:
            if seed_parameter.kind == inspect.Parameter.KEYWORD_ONLY:
                return "keyword_seed"
            if seed_parameter.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                return "two"

        if any(parameter.kind == inspect.Parameter.VAR_POSITIONAL for parameter in parameters.values()):
            return "two"
        return "one"

    def _evaluate(self, config: Dict[str, Any]) -> TrialResult:
        start = time.perf_counter()
        scores: List[float] = []
        try:
            for seed in self.seeds:
                self._set_seed(seed)
                if self._objective_mode == "keyword_seed":
                    value = self.objective(dict(config), seed=seed)
                elif self._objective_mode == "two":
                    value = self.objective(dict(config), seed)
                else:
                    value = self.objective(dict(config))
                try:
                    score = float(value)
                except (TypeError, ValueError) as exc:
                    raise TypeError("objective must return one numeric metric. Return a float or int.") from exc
                if not math.isfinite(score):
                    raise ValueError("objective returned a non-finite metric. Check the validation result.")
                scores.append(score)
        except Exception as exc:
            if not self.catch_exceptions:
                raise
            return TrialResult(
                config=dict(config),
                scores=tuple(scores),
                mean_score=float("inf") if self.direction == "min" else float("-inf"),
                std_score=float("inf"),
                selection_score=float("inf") if self.direction == "min" else float("-inf"),
                seeds=self.seeds[:len(scores)],
                error=(
                    f"{type(exc).__name__}: {exc}, seed={self.seeds[len(scores)]}, "
                    f"config={config!r}"
                ),
                duration=time.perf_counter() - start,
            )

        mean = sum(scores) / len(scores)
        variance = sum((score - mean) ** 2 for score in scores) / len(scores)
        std = math.sqrt(variance)
        adjustment = self.stability_penalty * std
        selection = mean + adjustment if self.direction == "min" else mean - adjustment

        return TrialResult(
            config=dict(config),
            scores=tuple(scores),
            mean_score=mean,
            std_score=std,
            selection_score=selection,
            seeds=self.seeds,
            duration=time.perf_counter() - start,
        )

    @staticmethod
    def _set_seed(seed: int) -> None:
        """Seed Python and optional ML libraries only if already imported.

        Avoids importing large frameworks just because this package is used.
        Users can seed additional libraries inside their objective callback.
        """
        random.seed(seed)

        numpy = sys.modules.get("numpy")
        if numpy is not None:
            numpy.random.seed(seed % (2 ** 32))

        torch = sys.modules.get("torch")
        if torch is not None:
            torch.manual_seed(seed)
            if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        tensorflow = sys.modules.get("tensorflow")
        if tensorflow is not None:
            try:
                tensorflow.random.set_seed(seed)
            except (AttributeError, RuntimeError):
                pass

    def _generate_candidates(
        self,
        count: int,
        cache: Mapping[Any, TrialResult],
        elites: Sequence[TrialResult],
    ) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        seen = set(cache.keys())
        attempts = 0
        max_attempts = max(100, count * 100)

        while len(candidates) < count and attempts < max_attempts:
            attempts += 1
            if elites and self._rng.random() >= self.exploration_mix:
                parent = self._select_parent(elites)
                candidate = self._generate_child(parent)
            else:
                candidate = self._generate_config()

            key = self._config_key(candidate)
            if key not in seen:
                candidates.append(candidate)
                seen.add(key)

        return candidates

    def _generate_config(self) -> Dict[str, Any]:
        config = {name: self._draw(name) for name in self.space}
        self._assert_valid_config(config)
        return config

    def _assert_valid_config(self, config: Mapping[str, Any]) -> None:
        assert set(config) == set(self.space), "OptiCan generated an incomplete configuration. Check parameter sampling."
        for name, spec in self.space.items():
            value = config[name]
            if isinstance(spec, Float):
                valid = (
                    value == 0 and spec.allow_zero
                ) or spec.low <= value <= spec.high
            elif isinstance(spec, Int):
                valid = (
                    isinstance(value, int)
                    and not isinstance(value, bool)
                    and spec.low <= value <= spec.high
                    and (value - spec.low) % spec.step == 0
                )
            elif isinstance(spec, Choice):
                valid = value in spec.values
            else:
                valid = True
            assert valid, "OptiCan generated a value outside its search space. Check parameter sampling."

    def _generate_child(self, parent: Mapping[str, Any]) -> Dict[str, Any]:
        child: Dict[str, Any] = {}
        changed = False
        for name in self.space:
            if self._rng.random() < self.mutation_rate:
                child[name] = self._draw(name, parent[name])
                changed = True
            else:
                child[name] = parent[name]

        if self.space and not changed:
            name = self._rng.choice(list(self.space))
            child[name] = self._draw(name, parent[name])
        self._assert_valid_config(child)
        return child

    def _draw(self, name: str, parent: Any = None) -> Any:
        spec = self.space[name]
        state = self._states.get(name, {})

        if isinstance(spec, Fixed):
            return spec.value

        if isinstance(spec, Float):
            if spec.allow_zero and self._rng.random() < state["zero_probability"]:
                return 0.0

            low, high = state["low"], state["high"]
            if parent is not None and parent != 0 and self._rng.random() < 0.8:
                center = math.log10(parent) if spec.log else float(parent)
                sigma = max((high - low) * 0.15, 1e-12)
                value = self._rng.gauss(center, sigma)
            else:
                value = self._rng.uniform(low, high)
            value = min(high, max(low, value))
            return 10 ** value if spec.log else float(value)

        if isinstance(spec, Int):
            low, high = state["low"], state["high"]
            if parent is not None and self._rng.random() < 0.8:
                center = float(parent)
                sigma = max((high - low) * 0.15, spec.step)
                value = self._rng.gauss(center, sigma)
            else:
                value = self._rng.uniform(low, high)
            value = min(high, max(low, value))
            steps = round((value - spec.low) / spec.step)
            candidate = spec.low + steps * spec.step
            max_steps = (spec.high - spec.low) // spec.step
            candidate = min(spec.low + max_steps * spec.step, max(spec.low, candidate))
            return int(candidate)

        if isinstance(spec, Choice):
            probabilities = state["probabilities"]
            index = self._weighted_index(probabilities)
            return spec.values[index]

        raise TypeError(f"Unsupported parameter type for {name!r}.")

    def _update_states(self, elites: Sequence[TrialResult]) -> None:
        for name, spec in self.space.items():
            values = [trial.config[name] for trial in elites]
            state = self._states.get(name)

            if isinstance(spec, Float):
                if spec.allow_zero:
                    zero_count = sum(value == 0 for value in values)
                    probability = (zero_count + 1) / (len(values) + 2)
                    state["zero_probability"] = min(0.75, max(0.05, probability))
                positive = [float(value) for value in values if value > 0]
                if positive:
                    transformed = [math.log10(v) if spec.log else v for v in positive]
                    state["low"], state["high"] = self._shrink(
                        state["low"], state["high"], transformed,
                        state["initial_low"], state["initial_high"],
                    )

            elif isinstance(spec, Int):
                state["low"], state["high"] = self._shrink(
                    state["low"], state["high"], [float(v) for v in values],
                    state["initial_low"], state["initial_high"],
                )

            elif isinstance(spec, Choice):
                counts = [1.0 + sum(value == choice for value in values) for choice in spec.values]
                total = sum(counts)
                observed = [count / total for count in counts]
                uniform = 1.0 / len(spec.values)
                state["probabilities"] = [
                    (1 - self.categorical_exploration) * p
                    + self.categorical_exploration * uniform
                    for p in observed
                ]
                assert all(p >= 0 for p in state["probabilities"]) and math.isclose(sum(state["probabilities"]), 1.0), "OptiCan created invalid category probabilities. Check elite frequency updates."

    def _shrink(
        self,
        low: float,
        high: float,
        samples: Sequence[float],
        initial_low: float,
        initial_high: float,
    ) -> Tuple[float, float]:
        if not samples or high <= low:
            return low, high

        initial_width = initial_high - initial_low
        minimum_width = initial_width * self.min_range_fraction
        current_width = high - low
        if current_width <= minimum_width:
            return low, high

        midpoint = (low + high) / 2
        left_count = sum(value <= midpoint for value in samples)
        right_count = len(samples) - left_count
        total = max(len(samples), 1)
        contraction = current_width * self.shrink_fraction

        left_contract = contraction * right_count / total
        right_contract = contraction * left_count / total
        q_low = self._percentile(samples, 10)
        q_high = self._percentile(samples, 90)
        margin = current_width * 0.02

        new_low = max(low + left_contract, q_low - margin)
        new_high = min(high - right_contract, q_high + margin)
        new_low = min(high, max(low, new_low))
        new_high = max(low, min(high, new_high))

        if new_high - new_low < minimum_width:
            center = (new_low + new_high) / 2
            new_low = center - minimum_width / 2
            new_high = center + minimum_width / 2
            new_low = max(initial_low, new_low)
            new_high = min(initial_high, new_high)

        if new_low >= new_high:
            return low, high
        assert initial_low <= new_low <= new_high <= initial_high, "OptiCan created invalid search bounds. Check numeric range shrinking."
        return new_low, new_high

    @staticmethod
    def _percentile(values: Sequence[float], percentile: float) -> float:
        ordered = sorted(float(value) for value in values)
        if len(ordered) == 1:
            return ordered[0]
        position = (len(ordered) - 1) * percentile / 100
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        fraction = position - lower
        return ordered[lower] * (1 - fraction) + ordered[upper] * fraction

    def _select_elites(self, results: Sequence[TrialResult]) -> List[TrialResult]:
        valid = [trial for trial in results if trial.successful]
        reverse = self.direction == "max"
        return sorted(
            valid,
            key=lambda trial: trial.selection_score,
            reverse=reverse,
        )[:self.elite_size]

    def _select_parent(self, elites: Sequence[TrialResult]) -> Dict[str, Any]:
        ranked = self._select_elites(elites)
        assert ranked, "OptiCan tried to select a parent without elite trials. Check elite selection."
        weights = [1.0 / ((rank + 1) ** 0.7) for rank in range(len(ranked))]
        return dict(self._rng.choices(ranked, weights=weights, k=1)[0].config)

    def _weighted_index(self, weights: Sequence[float]) -> int:
        assert weights and all(weight >= 0 for weight in weights), "OptiCan received invalid sampling weights. Check categorical probability updates."
        assert sum(weights) > 0, "OptiCan cannot sample from zero total probability. Check the search state."
        target = self._rng.random() * sum(weights)
        cumulative = 0.0
        for index, weight in enumerate(weights):
            cumulative += weight
            if target <= cumulative:
                return index
        return len(weights) - 1

    def _config_key(self, config: Mapping[str, Any]) -> Tuple[Any, ...]:
        return tuple((name, self._freeze(value)) for name, value in sorted(config.items()))

    @classmethod
    def _freeze(cls, value: Any) -> Any:
        if isinstance(value, Mapping):
            pairs = [(cls._freeze(k), cls._freeze(v)) for k, v in value.items()]
            return tuple(sorted(pairs, key=repr))
        if isinstance(value, (list, tuple)):
            return tuple(cls._freeze(item) for item in value)
        if isinstance(value, set):
            return tuple(sorted((cls._freeze(item) for item in value), key=repr))
        try:
            hash(value)
            return value
        except TypeError:
            return repr(value)

    def _improved(self, current: float, previous: float) -> bool:
        if self.direction == "min":
            return current < previous - self.min_delta
        return current > previous + self.min_delta
