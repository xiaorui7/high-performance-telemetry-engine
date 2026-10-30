"""Small internal model independent of any external dataset format."""
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class TelemetryChannel:
    name: str
    sampling_rate_hz: float
    values: np.ndarray
    unit: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TelemetryCycle:
    cycle_id: str
    channels: Mapping[str, TelemetryChannel]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AlignedBatch:
    cycle_ids: tuple[str, ...]
    channel_names: tuple[str, ...]
    values: np.ndarray
    sampling_rate_hz: float

