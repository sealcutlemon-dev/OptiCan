import math

import pytest

from optican import OptiCan, Bool, Choice, Float, Int, TuningResult


def test_minimizes_mixed_space_and_keeps_global_best():
    def objective(config, seed):
        return (
            (config["x"] - 0.25) ** 2
            + (config["depth"] - 4) ** 2 * 0.01
            + (0 if config["mode"] == "good" else 0.2)
        )

    tuner = OptiCan(
        objective,
        config={
            "x": Float(-1.0, 1.0),
            "depth": Int(1, 8),
            "mode": Choice(["bad", "good"]),
            "enabled": Bool(),
        },
        rounds=3,
        initial_candidates=32,
        candidates_per_round=12,
        seeds=2,
        verbose=False,
        seed=13,
    )
    best = tuner.optimize()

    assert isinstance(best, dict)
    assert tuner.result is not None
    assert best == tuner.result.last_round_best_config
    assert tuner.result.best_score <= min(
        trial.mean_score for trial in tuner.result.successful_trials
    ) + 1e-12
    assert len(tuner.result.successful_trials) > 0
    assert tuner.result.rounds_completed <= 3


def test_ubest_returns_full_result():
    tuner = OptiCan(
        lambda config: (config["x"] - 0.2) ** 2,
        config={"x": Float(0.0, 1.0)},
        rounds=2,
        initial_candidates=8,
        candidates_per_round=4,
        ubest=True,
        verbose=False,
    )
    result = tuner.optimize()

    assert isinstance(result, TuningResult)
    assert result.best_config == tuner.best_config
    assert result.summary()["successful_trials"] == len(result.successful_trials)


def test_maximization():
    tuner = OptiCan(
        lambda config: config["choice"],
        config={"choice": Choice([1, 2, 3, 4])},
        direction="max",
        rounds=2,
        initial_candidates=4,
        candidates_per_round=4,
        verbose=False,
    )
    best = tuner.optimize()
    assert best["choice"] == 4
    assert tuner.result.best_score == 4


def test_empty_config_runs_once():
    calls = []

    def objective(config):
        calls.append(config)
        return 1.25

    tuner = OptiCan(objective, config={}, rounds=3, verbose=False)
    assert tuner.optimize() == {}
    assert len(calls) == 1
    assert len(tuner.result.trials) == 1


def test_fixed_values_and_compact_specs():
    tuner = OptiCan(
        lambda config, seed: float(config["fixed"] == "ok") + config["depth"] * 0.0,
        config={
            "fixed": "ok",
            "depth": {"type": "int", "low": 2, "high": 2},
            "mode": ["a", "b"],
        },
        rounds=1,
        initial_candidates=3,
        verbose=False,
    )
    best = tuner.optimize()
    assert best["fixed"] == "ok"
    assert best["depth"] == 2
    assert best["mode"] in {"a", "b"}


def test_catches_failed_candidates_and_continues():
    def objective(config):
        if config["x"] < 0:
            raise RuntimeError("negative value is unsupported")
        return config["x"]

    tuner = OptiCan(
        objective,
        config={"x": Float(-1.0, 1.0)},
        rounds=2,
        initial_candidates=30,
        candidates_per_round=10,
        catch_exceptions=True,
        verbose=False,
        seed=8,
    )
    tuner.optimize()
    assert tuner.result.successful_trials
    assert any(not trial.successful for trial in tuner.result.trials)


def test_reproducible_candidate_sequence():
    def run():
        tuner = OptiCan(
            lambda config, seed: (config["x"] - 0.12) ** 2,
            config={"x": Float(-1, 1)},
            rounds=3,
            initial_candidates=10,
            candidates_per_round=5,
            verbose=False,
            seed=100,
        )
        tuner.optimize()
        return [trial.config for trial in tuner.result.trials]

    assert run() == run()


def test_invalid_options_are_rejected():
    with pytest.raises(ValueError):
        OptiCan(lambda config: 1, config={}, rounds=0)
    with pytest.raises(ValueError):
        OptiCan(lambda config: 1, config={}, direction="sideways")
    with pytest.raises(ValueError):
        Float(0, 1, log=True)
    with pytest.raises(ValueError):
        Int(1, 5, step=0)


def test_all_failed_raises_useful_error():
    tuner = OptiCan(
        lambda config: math.nan,
        config={"x": Choice([1, 2])},
        rounds=1,
        initial_candidates=2,
        verbose=False,
    )
    with pytest.raises(RuntimeError, match="All candidate evaluations failed"):
        tuner.optimize()


def test_reusing_tuner_resets_search_state():
    tuner = OptiCan(
        lambda config, seed: (config["x"] - 0.4) ** 2,
        config={"x": Float(-1, 1)},
        rounds=3,
        initial_candidates=12,
        candidates_per_round=5,
        seed=21,
        verbose=False,
    )
    first = tuner.optimize()
    second = tuner.optimize()
    assert first == second


def test_keyword_only_seed_callback():
    received = []

    def objective(config, *, seed):
        received.append(seed)
        return config["x"] ** 2

    tuner = OptiCan(
        objective,
        config={"x": Float(-1, 1)},
        rounds=1,
        initial_candidates=2,
        seeds=3,
        verbose=False,
    )
    tuner.optimize()
    assert received[:3] == [42, 43, 44]
    assert received[3:6] == [42, 43, 44]


def test_fixed_only_space_is_evaluated_once():
    calls = []

    def objective(config):
        calls.append(config)
        return 0.5

    tuner = OptiCan(
        objective,
        config={"fixed": "value", "single": Choice(["only"])},
        rounds=5,
        verbose=False,
    )
    assert tuner.optimize() == {"fixed": "value", "single": "only"}
    assert len(calls) == 1
    assert tuner.result.rounds_completed == 1


def test_float_string_bounds_are_normalized():
    spec = Float("0.001", "1.0")
    assert spec.low == 0.001
    assert spec.high == 1.0


def test_nan_search_options_are_rejected():
    with pytest.raises(ValueError):
        OptiCan(lambda config: 1.0, config={}, stability_penalty=float("nan"))
    with pytest.raises(ValueError):
        OptiCan(lambda config: 1.0, config={}, mutation_rate=float("nan"))


def test_ubest_requires_boolean():
    with pytest.raises(TypeError):
        OptiCan(lambda config: 1.0, config={}, ubest="yes")


def test_ubest_false_returns_latest_successful_round_best():
    scores = iter([0.1, 0.4, 0.9])
    tuner = OptiCan(
        lambda config: next(scores),
        config={"choice": Choice(["a", "b", "c"])},
        rounds=2,
        initial_candidates=2,
        candidates_per_round=1,
        exploration_mix=1.0,
        ubest=False,
        verbose=False,
        seed=5,
    )

    returned = tuner.optimize()

    assert returned == tuner.result.last_round_best_config
    assert tuner.result.last_round_best_score == 0.9
    assert tuner.result.best_score == 0.1


def test_ubest_true_returns_global_best_result():
    scores = iter([0.1, 0.4, 0.9])
    tuner = OptiCan(
        lambda config: next(scores),
        config={"choice": Choice(["a", "b", "c"])},
        rounds=2,
        initial_candidates=2,
        candidates_per_round=1,
        exploration_mix=1.0,
        ubest=True,
        verbose=False,
        seed=5,
    )

    returned = tuner.optimize()

    assert isinstance(returned, TuningResult)
    assert returned.best_score == 0.1
    assert returned.best_trial.mean_score == 0.1
    assert returned.last_round_best_score == 0.9


def test_objective_can_compute_a_custom_metric_formula():
    def objective(config):
        true_count = config["true_count"]
        all_count = config["all_count"]
        return true_count / all_count

    tuner = OptiCan(
        objective,
        config={"true_count": 8, "all_count": 10},
        direction="max",
        verbose=False,
    )

    assert tuner.optimize() == {"true_count": 8, "all_count": 10}
    assert tuner.result.best_score == 0.8


def test_optimizer_exposes_both_best_configurations():
    scores = iter([0.2, 0.3, 0.8])
    tuner = OptiCan(
        lambda config: next(scores),
        config={"choice": Choice(["a", "b", "c"])},
        rounds=2,
        initial_candidates=2,
        candidates_per_round=1,
        exploration_mix=1.0,
        verbose=False,
        seed=9,
    )
    last_round_config = tuner.optimize()

    assert tuner.last_round_best_config == last_round_config
    assert tuner.best_config == tuner.result.best_config
    assert tuner.result.best_score == 0.2


def test_developer_assertion_explains_invalid_sampler_output():
    tuner = OptiCan(
        lambda config: 1.0,
        config={"depth": Int(1, 3)},
        verbose=False,
    )
    tuner._draw = lambda name, parent=None: 99

    with pytest.raises(AssertionError, match="OptiCan generated a value outside its search space"):
        tuner._generate_config()


def test_callback_optional_non_seed_argument_is_not_overwritten():
    received = []

    def objective(config, metadata="keep"):
        received.append(metadata)
        return 1.0

    tuner = OptiCan(
        objective,
        config={},
        seeds=2,
        verbose=False,
    )
    tuner.optimize()

    assert received == ["keep", "keep"]
