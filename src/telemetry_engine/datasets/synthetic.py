"""Deterministic on-disk telemetry for tests, CI, and local benchmarks."""
import json
from pathlib import Path
from collections.abc import Iterator

import numpy as np

from ..model import TelemetryChannel, TelemetryCycle

DEFAULT_NAMES = ("pressure_1", "volume_flow_1", "temperature_1", "vibration_1", "motor_power")
DEFAULT_RATES = (100.0, 10.0, 1.0, 25.0, 50.0)
DEFAULT_UNITS = ("bar", "l/min", "degC", "mm/s", "W")
SCHEMA = "telemetry-synthetic-v1"


def generate_synthetic_dataset(path: str | Path, cycles: int = 100, channels: int = 5,
                               samples: int = 6000, sampling_rates: list[float] | None = None,
                               seed: int = 2026) -> Path:
    if type(cycles) is not int or cycles < 1 or type(channels) is not int or channels < 1:
        raise ValueError("cycles and channels must be positive integers")
    if type(samples) is not int or samples < 2:
        raise ValueError("samples must be an integer >= 2")
    rates = list(sampling_rates or DEFAULT_RATES[:channels])
    if len(rates) != channels or any(not np.isfinite(rate) or rate <= 0 for rate in rates):
        raise ValueError("provide one positive finite sampling rate per channel")
    names = [DEFAULT_NAMES[i] if i < len(DEFAULT_NAMES) else f"sensor_{i + 1}" for i in range(channels)]
    units = [DEFAULT_UNITS[i] if i < len(DEFAULT_UNITS) else None for i in range(channels)]
    highest_rate = max(rates)
    duration = samples / highest_rate
    output = Path(path)
    cycle_dir = output / "cycles"
    cycle_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": SCHEMA, "cycles": cycles, "seed": seed, "duration_seconds": duration,
        "channels": [{"name": name, "sampling_rate_hz": rate, "unit": unit}
                     for name, rate, unit in zip(names, rates, units)],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    rng = np.random.default_rng(seed)
    for cycle_index in range(cycles):
        arrays = {}
        drift = cycle_index / max(cycles - 1, 1)
        for channel_index, (name, rate) in enumerate(zip(names, rates)):
            count = max(1, int(round(duration * rate)))
            time_axis = np.arange(count) / rate
            baseline = (channel_index + 1) * 10 + drift
            signal = baseline + (1 + .2 * channel_index) * np.sin(
                2 * np.pi * (0.05 + channel_index * 0.015) * time_axis + cycle_index * .01
            )
            noise = rng.normal(0, .03 * (channel_index + 1), count)
            arrays[name] = (signal + noise).astype(np.float64)
        np.savez_compressed(cycle_dir / f"cycle_{cycle_index:06d}.npz", **arrays)
    return output


class SyntheticDataset:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        try:
            self.manifest = json.loads((self.path / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read synthetic manifest: {exc}") from exc
        if self.manifest.get("schema") != SCHEMA:
            raise ValueError("unsupported synthetic dataset schema")
        if not isinstance(self.manifest.get("channels"), list) or not self.manifest["channels"]:
            raise ValueError("synthetic manifest has no channels")

    @property
    def source_name(self) -> str:
        return "synthetic"

    def iter_cycles(self) -> Iterator[TelemetryCycle]:
        count = self.manifest.get("cycles")
        if type(count) is not int or count < 1:
            raise ValueError("synthetic manifest has invalid cycle count")
        for index in range(count):
            path = self.path / "cycles" / f"cycle_{index:06d}.npz"
            try:
                with np.load(path, allow_pickle=False) as archive:
                    channels = {}
                    expected = {item["name"] for item in self.manifest["channels"]}
                    if set(archive.files) != expected:
                        raise ValueError(f"{path.name}: channel fields do not match manifest")
                    for item in self.manifest["channels"]:
                        name = item["name"]
                        channels[name] = TelemetryChannel(
                            name, float(item["sampling_rate_hz"]), np.asarray(archive[name], dtype=np.float64),
                            item.get("unit"), {"source": "synthetic"},
                        )
            except (OSError, EOFError, ValueError) as exc:
                if isinstance(exc, ValueError) and "channel fields" in str(exc):
                    raise
                raise ValueError(f"cannot read synthetic cycle {path}: {exc}") from exc
            yield TelemetryCycle(f"synthetic-{index:06d}", channels, {"source": "synthetic", "index": index})

