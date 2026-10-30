"""Explicit validation at dataset and processing boundaries."""
from collections.abc import Iterable
import math

import numpy as np

from .config import EngineConfig
from .model import TelemetryChannel, TelemetryCycle


def validate_channel(channel: TelemetryChannel, limits: tuple[float | None, float | None] | None = None) -> None:
    if not isinstance(channel.name, str) or not channel.name.strip():
        raise ValueError("channel name must be nonempty")
    if not math.isfinite(channel.sampling_rate_hz) or channel.sampling_rate_hz <= 0:
        raise ValueError(f"channel {channel.name}: sampling rate must be finite and positive")
    values = np.asarray(channel.values)
    if values.ndim != 1:
        raise ValueError(f"channel {channel.name}: values must be one-dimensional")
    if values.size == 0:
        raise ValueError(f"channel {channel.name}: values cannot be empty")
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError(f"channel {channel.name}: values must be numeric")
    if not np.isfinite(values).all():
        raise ValueError(f"channel {channel.name}: values contain NaN or infinity")
    if limits:
        low, high = limits
        if low is not None and np.any(values < low):
            raise ValueError(f"channel {channel.name}: value below configured minimum {low}")
        if high is not None and np.any(values > high):
            raise ValueError(f"channel {channel.name}: value above configured maximum {high}")


def validate_cycle(cycle: TelemetryCycle, config: EngineConfig) -> None:
    if not isinstance(cycle.cycle_id, str) or not cycle.cycle_id.strip():
        raise ValueError("cycle_id must be nonempty")
    if not cycle.channels:
        raise ValueError(f"cycle {cycle.cycle_id}: no channels")
    missing = set(config.required_channels).difference(cycle.channels)
    if missing:
        raise ValueError(f"cycle {cycle.cycle_id}: missing required channels: {', '.join(sorted(missing))}")
    durations = []
    for key, channel in cycle.channels.items():
        if key != channel.name:
            raise ValueError(f"cycle {cycle.cycle_id}: channel key/name mismatch for {key}")
        validate_channel(channel, config.value_ranges.get(key))
        durations.append(len(channel.values) / channel.sampling_rate_hz)
    tolerance = max(1 / channel.sampling_rate_hz for channel in cycle.channels.values())
    if max(durations) - min(durations) > tolerance + 1e-12:
        raise ValueError(f"cycle {cycle.cycle_id}: channel durations are inconsistent")


def validate_cycles(cycles: Iterable[TelemetryCycle], config: EngineConfig) -> int:
    seen: set[str] = set()
    count = 0
    for cycle in cycles:
        validate_cycle(cycle, config)
        if cycle.cycle_id in seen:
            raise ValueError(f"duplicate cycle ID: {cycle.cycle_id}")
        seen.add(cycle.cycle_id)
        count += 1
    if count == 0:
        raise ValueError("input contains no telemetry cycles")
    return count

