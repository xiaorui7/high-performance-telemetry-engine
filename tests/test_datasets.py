import numpy as np
import pytest

from telemetry_engine.datasets.hydraulic import HydraulicDataset
from telemetry_engine.datasets.source import open_dataset
from telemetry_engine.datasets.synthetic import SyntheticDataset, generate_synthetic_dataset


def test_synthetic_is_deterministic(tmp_path):
    one = SyntheticDataset(generate_synthetic_dataset(tmp_path / "one", 2, 2, 20, [4, 2], 10))
    two = SyntheticDataset(generate_synthetic_dataset(tmp_path / "two", 2, 2, 20, [4, 2], 10))
    first = next(one.iter_cycles())
    second = next(two.iter_cycles())
    for name in first.channels:
        np.testing.assert_array_equal(first.channels[name].values, second.channels[name].values)


def test_synthetic_missing_cycle(tmp_path):
    root = generate_synthetic_dataset(tmp_path / "data", 1, 1, 10, [2])
    (root / "cycles" / "cycle_000000.npz").unlink()
    with pytest.raises(ValueError, match="cannot read synthetic cycle"):
        list(SyntheticDataset(root).iter_cycles())


def test_malformed_manifest(tmp_path):
    root = tmp_path / "bad"
    root.mkdir()
    (root / "manifest.json").write_text("{")
    with pytest.raises(ValueError, match="manifest"):
        SyntheticDataset(root)


def test_hydraulic_adapter_tiny_fixture(tmp_path):
    (tmp_path / "PS1.txt").write_text(" ".join(["1"] * 6000) + "\n")
    (tmp_path / "FS1.txt").write_text(" ".join(["2"] * 600) + "\n")
    (tmp_path / "profile.txt").write_text("100 100 0 130 0\n")
    dataset = HydraulicDataset(tmp_path, ("pressure_1", "volume_flow_1"))
    cycle = next(dataset.iter_cycles())
    assert cycle.cycle_id == "hydraulic-0001"
    assert len(cycle.channels["pressure_1"].values) == 6000
    assert cycle.metadata["condition_profile"]["pump_leakage"] == 0


def test_hydraulic_rejects_bad_row(tmp_path):
    (tmp_path / "PS1.txt").write_text("1 2\n")
    with pytest.raises(ValueError, match="expected 6000"):
        list(HydraulicDataset(tmp_path, ("pressure_1",)).iter_cycles())


def test_source_detection(tmp_path):
    root = generate_synthetic_dataset(tmp_path / "data", 1, 1, 10, [2])
    assert open_dataset(root).source_name == "synthetic"

