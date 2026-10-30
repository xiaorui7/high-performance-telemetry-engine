"""Align heterogeneous channel rates to a common per-cycle time grid."""
import numpy as np

from .config import EngineConfig
from .model import AlignedBatch, TelemetryCycle
from .validation import validate_cycle


def _resample(values: np.ndarray, source_rate: float, target_time: np.ndarray, method: str) -> np.ndarray:
    source_time = np.arange(len(values), dtype=np.float64) / source_rate
    if method == "linear":
        return np.interp(target_time, source_time, values).astype(np.float64, copy=False)
    indices = np.rint(target_time * source_rate).astype(np.int64)
    return np.asarray(values, dtype=np.float64)[np.minimum(indices, len(values) - 1)]


def align_cycles(cycles: list[TelemetryCycle], config: EngineConfig,
                 channel_names: tuple[str, ...] | None = None) -> AlignedBatch:
    if not cycles:
        raise ValueError("cannot align an empty batch")
    for cycle in cycles:
        validate_cycle(cycle, config)
    names = channel_names or config.required_channels or tuple(cycles[0].channels)
    if not names:
        raise ValueError("no channels selected for alignment")
    for cycle in cycles:
        missing = set(names).difference(cycle.channels)
        extra = set(cycle.channels).difference(names)
        if missing or (not config.required_channels and extra):
            raise ValueError(f"cycle {cycle.cycle_id}: inconsistent channel set")
    duration = min(
        len(cycle.channels[name].values) / cycle.channels[name].sampling_rate_hz
        for cycle in cycles for name in names
    )
    sample_count = int(np.floor(duration * config.target_rate_hz + 1e-12))
    if sample_count < 1:
        raise ValueError("target rate and cycle duration produce no aligned samples")
    target_time = np.arange(sample_count, dtype=np.float64) / config.target_rate_hz
    values = np.empty((len(cycles), len(names), sample_count), dtype=np.float64)
    for cycle_index, cycle in enumerate(cycles):
        for channel_index, name in enumerate(names):
            channel = cycle.channels[name]
            values[cycle_index, channel_index] = _resample(
                np.asarray(channel.values, dtype=np.float64), channel.sampling_rate_hz,
                target_time, config.alignment,
            )
    return AlignedBatch(tuple(c.cycle_id for c in cycles), tuple(names), values, config.target_rate_hz)
