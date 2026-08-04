"""Small, dependency-light statistics for paired benchmark evidence."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import sqrt

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Interval:
    """Closed confidence interval."""

    low: float
    high: float
    confidence: float


@dataclass(frozen=True, slots=True)
class MetricSummary:
    """Descriptive statistics for one metric and method."""

    count: int
    mean: float
    median: float
    standard_deviation: float
    standard_error: float
    mean_interval: Interval


@dataclass(frozen=True, slots=True)
class PairedDifference:
    """Paired ``candidate - baseline`` comparison over shared event tapes."""

    count: int
    mean_difference: float
    median_difference: float
    improvement_probability: float
    mean_interval: Interval


def _as_nonempty(values: Iterable[float], name: str) -> FloatArray:
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def bootstrap_mean_interval(
    values: Iterable[float],
    *,
    confidence: float = 0.95,
    resamples: int = 2_000,
    seed: int = 0,
) -> Interval:
    """Return a percentile bootstrap interval for the arithmetic mean."""

    array = _as_nonempty(values, "values")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    if resamples < 100:
        raise ValueError("resamples must be at least 100")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(resamples, array.size))
    means = array[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [tail, 1.0 - tail])
    return Interval(float(low), float(high), confidence)


def summarize(
    values: Iterable[float],
    *,
    confidence: float = 0.95,
    resamples: int = 2_000,
    seed: int = 0,
) -> MetricSummary:
    """Summarize one metric without hiding its sample count or uncertainty."""

    array = _as_nonempty(values, "values")
    deviation = float(array.std(ddof=1)) if array.size > 1 else 0.0
    return MetricSummary(
        count=int(array.size),
        mean=float(array.mean()),
        median=float(np.median(array)),
        standard_deviation=deviation,
        standard_error=deviation / sqrt(array.size),
        mean_interval=bootstrap_mean_interval(
            array,
            confidence=confidence,
            resamples=resamples,
            seed=seed,
        ),
    )


def compare_paired(
    candidate: Iterable[float],
    baseline: Iterable[float],
    *,
    higher_is_better: bool = True,
    confidence: float = 0.95,
    resamples: int = 2_000,
    seed: int = 0,
) -> PairedDifference:
    """Compare methods evaluated on identically ordered seeds and event tapes."""

    candidate_array = _as_nonempty(candidate, "candidate")
    baseline_array = _as_nonempty(baseline, "baseline")
    if candidate_array.shape != baseline_array.shape:
        raise ValueError("paired samples must have identical lengths")
    raw_difference = candidate_array - baseline_array
    oriented = raw_difference if higher_is_better else -raw_difference
    interval = bootstrap_mean_interval(
        raw_difference,
        confidence=confidence,
        resamples=resamples,
        seed=seed,
    )
    wins = np.count_nonzero(oriented > 0.0)
    ties = np.count_nonzero(oriented == 0.0)
    probability = (wins + 0.5 * ties) / oriented.size
    return PairedDifference(
        count=int(oriented.size),
        mean_difference=float(raw_difference.mean()),
        median_difference=float(np.median(raw_difference)),
        improvement_probability=float(probability),
        mean_interval=interval,
    )
