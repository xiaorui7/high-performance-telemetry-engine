import numpy as np
import numba
import pytest

from telemetry_engine.alignment import align_cycles
from telemetry_engine.config import EngineConfig
from telemetry_engine.features import BACKENDS, FEATURE_NAMES, reference_features
from telemetry_engine.model import TelemetryChannel, TelemetryCycle


def multirate_cycle():
    return TelemetryCycle("cycle-1", {
        "fast": TelemetryChannel("fast", 4, np.array([0., 1., 2., 3., 4., 5., 6., 7.])),
        "slow": TelemetryChannel("slow", 1, np.array([0., 4.])),
    })


def test_linear_alignment():
    aligned = align_cycles([multirate_cycle()], EngineConfig(target_rate_hz=2, alignment="linear"))
    assert aligned.values.shape == (1, 2, 4)
    np.testing.assert_allclose(aligned.values[0, 0], [0, 2, 4, 6])
    np.testing.assert_allclose(aligned.values[0, 1], [0, 2, 4, 4])


def test_nearest_alignment():
    aligned = align_cycles([multirate_cycle()], EngineConfig(target_rate_hz=2, alignment="nearest"))
    np.testing.assert_array_equal(aligned.values[0, 1], [0, 0, 4, 4])


def test_inconsistent_channel_set_rejected():
    other = TelemetryCycle("cycle-2", {"fast": multirate_cycle().channels["fast"]})
    with pytest.raises(ValueError, match="inconsistent channel set"):
        align_cycles([multirate_cycle(), other], EngineConfig())


def test_reference_features_known_values():
    values = np.array([[[1., 2., 3., 4.]]])
    result = reference_features(values)[0, 0]
    expected = [2.5, np.std(values), 1, 4, np.sqrt(7.5), 3, 2.5, 3.85]
    np.testing.assert_allclose(result, expected)
    assert FEATURE_NAMES[4] == "rms"


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_backend_agreement(backend):
    numba.set_num_threads(min(2, numba.config.NUMBA_NUM_THREADS))
    values = np.random.default_rng(7).normal(size=(7, 4, 41))
    expected = BACKENDS["reference"](values)
    actual = BACKENDS[backend](values)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_backend_rejects_nonfinite():
    with pytest.raises(ValueError, match="finite"):
        BACKENDS["numpy"](np.array([[[np.nan]]]))

