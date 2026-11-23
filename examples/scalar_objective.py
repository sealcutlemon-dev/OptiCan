"""Small dependency-free example of the search algorithm."""

from optican import OptiCan, Choice, Float, Int


def objective(config, seed):
    return (config["x"] - 0.3) ** 2 + (config["y"] - 7) ** 2 * 0.002 + (
        0.0 if config["mode"] == "b" else 0.2
    )


if __name__ == "__main__":
    tuner = OptiCan(
        objective,
        config={
            "x": Float(-2.0, 2.0),
            "y": Int(1, 15),
            "mode": Choice(["a", "b", "c"]),
        },
        rounds=3,
        initial_candidates=30,
        candidates_per_round=16,
        seeds=1,
        ubest=True,
    )
    result = tuner.optimize()
    print(result.summary())
