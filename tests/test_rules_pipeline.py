import csv
import json

import numpy as np
import pytest

from telemetry_engine.config import EngineConfig, RuleConfig
from telemetry_engine.datasets.synthetic import SyntheticDataset, generate_synthetic_dataset
from telemetry_engine.pipeline import batched, process_cycles, process_to_directory
from telemetry_engine.rules import apply_rules


def test_rule_flags_high_value():
    features = np.zeros((1, 8))
    features[0, 4] = 2.0
    flags = apply_rules(features, ("vibration",), {"vibration_rms": RuleConfig(maximum=1, flag="HIGH_VIBRATION")})
    assert flags == ["HIGH_VIBRATION"]


def test_invalid_rule_key():
    with pytest.raises(ValueError, match="<channel>_<feature>"):
        apply_rules(np.zeros((1, 8)), ("a",), {"bad": RuleConfig(maximum=1)})


def test_batched_sizes():
    assert [len(group) for group in batched(range(7), 3)] == [3, 3, 1]


def test_process_backends_and_batching(tmp_path):
    dataset_path = generate_synthetic_dataset(tmp_path / "data", cycles=5, channels=2, samples=40,
                                              sampling_rates=[4, 2])
    dataset = SyntheticDataset(dataset_path)
    config = EngineConfig(target_rate_hz=2, batch_size=2)
    reference = process_cycles(dataset.iter_cycles(), config, "reference")
    numpy_result = process_cycles(dataset.iter_cycles(), config, "numpy")
    assert len(reference) == 5
    assert reference[0]["features"].keys() == numpy_result[0]["features"].keys()
    np.testing.assert_allclose(list(reference[0]["features"].values()), list(numpy_result[0]["features"].values()))


def test_structured_outputs(tmp_path):
    dataset = SyntheticDataset(generate_synthetic_dataset(tmp_path / "data", cycles=3, channels=2,
                                                          samples=40, sampling_rates=[4, 2]))
    output = tmp_path / "results"
    summary = process_to_directory(dataset.iter_cycles(), EngineConfig(target_rate_hz=2, batch_size=2),
                                   output, "numpy", input_source="test")
    assert summary["cycles_processed"] == 3
    assert len(list(csv.DictReader((output / "features.csv").open()))) == 3
    assert len(json.loads((output / "flags.json").read_text())) == 3
    metadata = json.loads((output / "run_metadata.json").read_text())
    assert metadata["backend"] == "numpy" and metadata["batch_size"] == 2

