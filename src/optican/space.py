from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
from collections.abc import Mapping
from typing import Any, Iterable, Tuple


@dataclass(frozen=True)
class Float:
    """A continuous parameter.

    Args:
        low: Inclusive lower bound. Must be positive when ``log=True``.
        high: Inclusive upper bound.
        log: Sample and adapt in log10 space.
        allow_zero: Also allow an exact zero value as a separate option.
        zero_probability: Initial probability of sampling exact zero.
    """

    low: float
    high: float
    log: bool = False
    allow_zero: bool = False
    zero_probability: float = 0.15

    def __post_init__(self) -> None:
        if isinstance(self.low, bool) or isinstance(self.high, bool):
            raise TypeError("Float bounds must be numbers, not booleans.")
        try:
            low, high = float(self.low), float(self.high)
        except (TypeError, ValueError) as exc:
            raise TypeError("Float bounds must be numeric.") from exc
        if not math.isfinite(low) or not math.isfinite(high):
            raise ValueError("Float bounds must be finite.")
        if low >= high:
            raise ValueError("Float requires low < high.")
        if not isinstance(self.log, bool) or not isinstance(self.allow_zero, bool):
            raise TypeError("log and allow_zero must be booleans.")
        if self.log and low <= 0:
            raise ValueError("Float with log=True requires low > 0.")
        if isinstance(self.zero_probability, bool):
            raise TypeError("zero_probability must be numeric.")
        try:
            probability = float(self.zero_probability)
        except (TypeError, ValueError) as exc:
            raise TypeError("zero_probability must be numeric.") from exc
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("zero_probability must be finite and between 0 and 1.")
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)
        object.__setattr__(self, "zero_probability", probability)


@dataclass(frozen=True)
class Int:
    """An integer parameter sampled on an inclusive stepped range."""

    low: int
    high: int
    step: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.low, bool) or isinstance(self.high, bool):
            raise TypeError("Int bounds must be integers, not booleans.")
        if not isinstance(self.low, Integral) or not isinstance(self.high, Integral):
            raise TypeError("Int bounds must be integers.")
        if self.low > self.high:
            raise ValueError("Int requires low <= high.")
        if isinstance(self.step, bool) or not isinstance(self.step, Integral):
            raise TypeError("Int step must be an integer.")
        if self.step < 1:
            raise ValueError("Int step must be >= 1.")
        object.__setattr__(self, "low", int(self.low))
        object.__setattr__(self, "high", int(self.high))
        object.__setattr__(self, "step", int(self.step))


@dataclass(frozen=True)
class Choice:
    """A categorical parameter with one or more possible values."""

    values: Tuple[Any, ...]

    def __init__(self, values: Iterable[Any]):
        if isinstance(values, set):
            values = sorted(values, key=repr)
        values = tuple(values)
        if not values:
            raise ValueError("Choice requires at least one value.")
        seen = set()
        for value in values:
            try:
                hash(value)
            except TypeError as exc:
                raise TypeError("Choice values must be hashable.") from exc
            if value in seen:
                raise ValueError("Choice values must be unique.")
            seen.add(value)
        object.__setattr__(self, "values", values)


@dataclass(frozen=True)
class Bool(Choice):
    """A boolean parameter, equivalent to ``Choice([False, True])``."""

    def __init__(self):
        object.__setattr__(self, "values", (False, True))


@dataclass(frozen=True)
class Fixed:
    """A fixed value that is passed through unchanged."""

    value: Any


Parameter = Float | Int | Choice | Fixed


def normalize_parameter(value: Any) -> Parameter:
    """Convert convenient config values and dictionaries into parameter specs.

    Supported forms:
      * ``Float(...)``, ``Int(...)``, ``Choice(...)``, ``Bool()``, ``Fixed(...)``
      * ``[1, 2, 3]`` or ``("a", "b")`` for categorical choices
      * dictionaries using ``type`` and the matching fields
      * any other scalar as a fixed value
    """
    if isinstance(value, (Float, Int, Choice, Fixed)):
        return value

    if isinstance(value, Mapping):
        kind = str(value.get("type", "")).lower()
        if kind in {"float", "continuous"}:
            scale = value.get("scale")
            if scale is not None and scale not in {"linear", "log"}:
                raise ValueError("Float scale must be 'linear' or 'log'.")
            return Float(
                low=value["low"],
                high=value["high"],
                log=value.get("log", scale == "log"),
                allow_zero=value.get("allow_zero", False),
                zero_probability=value.get("zero_probability", 0.15),
            )
        if kind in {"int", "integer"}:
            return Int(value["low"], value["high"], value.get("step", 1))
        if kind in {"choice", "categorical", "bool", "boolean"}:
            if kind in {"bool", "boolean"}:
                return Bool()
            return Choice(value["values"])
        if kind == "fixed":
            return Fixed(value["value"])
        raise ValueError(
            "Dictionary parameter specs require type='float', 'int', "
            "'choice', 'bool', or 'fixed'."
        )

    if isinstance(value, (list, tuple, set, range)):
        return Choice(value)

    return Fixed(value)
