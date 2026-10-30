"""Numerically equivalent reference, NumPy, and parallel Numba feature kernels."""
import math

import numpy as np
from numba import njit, prange

FEATURE_NAMES = ("mean", "std", "min", "max", "rms", "peak_to_peak", "median", "p95")


def _validate_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 3 or min(values.shape) < 1:
        raise ValueError("aligned values must have shape (cycles, channels, samples)")
    if not np.isfinite(values).all():
        raise ValueError("aligned values must be finite")
    return np.ascontiguousarray(values)


def _linear_quantile(sorted_values: list[float], quantile: float) -> float:
    position = (len(sorted_values) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def reference_features(values: np.ndarray) -> np.ndarray:
    """Readable scalar oracle; explicit loops intentionally favor traceability."""
    values = _validate_values(values)
    result = np.empty((values.shape[0], values.shape[1], len(FEATURE_NAMES)), dtype=np.float64)
    for cycle in range(values.shape[0]):
        for channel in range(values.shape[1]):
            row = values[cycle, channel]
            total = 0.0
            square_total = 0.0
            minimum = maximum = float(row[0])
            copied: list[float] = []
            for raw in row:
                value = float(raw)
                copied.append(value)
                total += value
                square_total += value * value
                minimum = min(minimum, value)
                maximum = max(maximum, value)
            mean = total / len(row)
            variance = 0.0
            for value in copied:
                variance += (value - mean) ** 2
            copied.sort()
            result[cycle, channel] = (
                mean, math.sqrt(variance / len(row)), minimum, maximum,
                math.sqrt(square_total / len(row)), maximum - minimum,
                _linear_quantile(copied, .5), _linear_quantile(copied, .95),
            )
    return result


def numpy_features(values: np.ndarray) -> np.ndarray:
    values = _validate_values(values)
    return np.stack((
        np.mean(values, axis=2), np.std(values, axis=2), np.min(values, axis=2),
        np.max(values, axis=2), np.sqrt(np.mean(values * values, axis=2)),
        np.ptp(values, axis=2), np.median(values, axis=2), np.percentile(values, 95, axis=2),
    ), axis=2)


@njit(cache=True, parallel=True)
def _numba_kernel(values: np.ndarray) -> np.ndarray:
    cycles, channels, samples = values.shape
    result = np.empty((cycles, channels, 8), dtype=np.float64)
    for index in prange(cycles * channels):
        cycle = index // channels
        channel = index % channels
        total = 0.0
        square_total = 0.0
        minimum = values[cycle, channel, 0]
        maximum = minimum
        for sample in range(samples):
            value = values[cycle, channel, sample]
            total += value
            square_total += value * value
            if value < minimum:
                minimum = value
            if value > maximum:
                maximum = value
        mean = total / samples
        variance = 0.0
        for sample in range(samples):
            difference = values[cycle, channel, sample] - mean
            variance += difference * difference
        ordered = np.sort(values[cycle, channel].copy())
        middle = (samples - 1) * .5
        med_low = int(math.floor(middle))
        med_high = int(math.ceil(middle))
        p95_position = (samples - 1) * .95
        p95_low = int(math.floor(p95_position))
        p95_high = int(math.ceil(p95_position))
        result[cycle, channel, 0] = mean
        result[cycle, channel, 1] = math.sqrt(variance / samples)
        result[cycle, channel, 2] = minimum
        result[cycle, channel, 3] = maximum
        result[cycle, channel, 4] = math.sqrt(square_total / samples)
        result[cycle, channel, 5] = maximum - minimum
        result[cycle, channel, 6] = ordered[med_low] * (med_high - middle) + ordered[med_high] * (middle - med_low) if med_low != med_high else ordered[med_low]
        result[cycle, channel, 7] = ordered[p95_low] * (p95_high - p95_position) + ordered[p95_high] * (p95_position - p95_low) if p95_low != p95_high else ordered[p95_low]
    return result


def numba_features(values: np.ndarray) -> np.ndarray:
    return _numba_kernel(_validate_values(values))


BACKENDS = {
    "reference": reference_features,
    "numpy": numpy_features,
    "numba": numba_features,
}


def get_backend(name: str):
    try:
        return BACKENDS[name]
    except KeyError as exc:
        raise ValueError(f"unknown backend: {name}") from exc

