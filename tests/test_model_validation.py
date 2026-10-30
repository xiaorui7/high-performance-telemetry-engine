import numpy as np
import pytest

from telemetry_engine.config import EngineConfig
from telemetry_engine.model import TelemetryChannel, TelemetryCycle
from telemetry_engine.validation import validate_channel, validate_cycle, validate_cycles


def channel(name="pressure", values=None, rate=2.0):
    return TelemetryChannel(name, rate, np.array([1., 2., 3., 4.]) if values is None else np.asarray(values))


def cycle(identifier="c1"):
    item = channel()
    return TelemetryCycle(identifier, {item.name: item})


def test_valid_cycle():
    validate_cycle(cycle(), EngineConfig(required_channels=("pressure",)))


@pytest.mark.parametrize("values,match", [([], "empty"), ([1, np.nan], "NaN"), ([1, np.inf], "infinity")])
def test_invalid_values(values, match):
    with pytest.raises(ValueError, match=match):
        validate_channel(channel(values=values))


@pytest.mark.parametrize("rate", [0, -1, np.inf])
def test_invalid_sampling_rate(rate):
    with pytest.raises(ValueError, match="sampling rate"):
        validate_channel(channel(rate=rate))


def test_invalid_dimensions():
    with pytest.raises(ValueError, match="one-dimensional"):
        validate_channel(channel(values=np.ones((2, 2))))


def test_missing_required_channel():
    with pytest.raises(ValueError, match="missing required"):
        validate_cycle(cycle(), EngineConfig(required_channels=("temperature",)))


def test_duplicate_cycle_id():
    with pytest.raises(ValueError, match="duplicate"):
        validate_cycles([cycle(), cycle()], EngineConfig())


def test_configured_value_range():
    with pytest.raises(ValueError, match="maximum"):
        validate_cycle(cycle(), EngineConfig(value_ranges={"pressure": (0, 3)}))


def test_inconsistent_duration():
    first = channel("a", np.ones(10), 10)
    second = channel("b", np.ones(30), 10)
    with pytest.raises(ValueError, match="durations"):
        validate_cycle(TelemetryCycle("c", {"a": first, "b": second}), EngineConfig())

