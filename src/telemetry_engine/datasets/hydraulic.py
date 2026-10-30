"""Streaming adapter for UCI Condition Monitoring of Hydraulic Systems."""
from contextlib import ExitStack
from itertools import zip_longest
from pathlib import Path
from collections.abc import Iterator

import numpy as np

from ..model import TelemetryChannel, TelemetryCycle

SENSORS = {
    "pressure_1": ("PS1.txt", 100.0, "bar"),
    "pressure_2": ("PS2.txt", 100.0, "bar"),
    "pressure_3": ("PS3.txt", 100.0, "bar"),
    "pressure_4": ("PS4.txt", 100.0, "bar"),
    "pressure_5": ("PS5.txt", 100.0, "bar"),
    "pressure_6": ("PS6.txt", 100.0, "bar"),
    "motor_power": ("EPS1.txt", 100.0, "W"),
    "volume_flow_1": ("FS1.txt", 10.0, "l/min"),
    "volume_flow_2": ("FS2.txt", 10.0, "l/min"),
    "temperature_1": ("TS1.txt", 1.0, "degC"),
    "temperature_2": ("TS2.txt", 1.0, "degC"),
    "temperature_3": ("TS3.txt", 1.0, "degC"),
    "temperature_4": ("TS4.txt", 1.0, "degC"),
    "vibration_1": ("VS1.txt", 1.0, "mm/s"),
    "cooling_efficiency": ("CE.txt", 1.0, "%"),
    "cooling_power": ("CP.txt", 1.0, "kW"),
    "efficiency_factor": ("SE.txt", 1.0, "%"),
}
DEFAULT_CHANNELS = ("pressure_1", "volume_flow_1", "temperature_1", "vibration_1", "motor_power")
PROFILE_FIELDS = ("cooler_condition_percent", "valve_condition_percent", "pump_leakage",
                  "accumulator_pressure_bar", "stable_flag")


class HydraulicDataset:
    """Read one row per sensor at a time; the 73 MB archive is never committed or loaded whole."""
    def __init__(self, path: str | Path, channels: tuple[str, ...] = DEFAULT_CHANNELS):
        self.path = Path(path)
        if not channels or len(set(channels)) != len(channels):
            raise ValueError("hydraulic channels must be nonempty and unique")
        unknown = set(channels).difference(SENSORS)
        if unknown:
            raise ValueError(f"unknown hydraulic channels: {', '.join(sorted(unknown))}")
        self.channels = tuple(channels)
        missing = [SENSORS[name][0] for name in self.channels if not (self.path / SENSORS[name][0]).is_file()]
        if missing:
            raise ValueError(f"missing UCI hydraulic files: {', '.join(missing)}")

    @property
    def source_name(self) -> str:
        return "uci-hydraulic"

    def iter_cycles(self) -> Iterator[TelemetryCycle]:
        profile_path = self.path / "profile.txt"
        with ExitStack() as stack:
            streams = [stack.enter_context((self.path / SENSORS[name][0]).open(encoding="utf-8"))
                       for name in self.channels]
            profile_stream = stack.enter_context(profile_path.open(encoding="utf-8")) if profile_path.is_file() else None
            rows = streams + ([profile_stream] if profile_stream else [])
            for index, lines in enumerate(zip_longest(*rows), start=1):
                if any(line is None for line in lines):
                    raise ValueError(f"UCI files have inconsistent row counts near cycle {index}")
                channels = {}
                for name, line in zip(self.channels, lines[:len(self.channels)]):
                    filename, rate, unit = SENSORS[name]
                    values = np.fromstring(line, sep=" ", dtype=np.float64)
                    expected = int(rate * 60)
                    if values.size != expected:
                        raise ValueError(f"{filename} cycle {index}: expected {expected} values, found {values.size}")
                    channels[name] = TelemetryChannel(
                        name, rate, values, unit,
                        {"source": "UCI Condition Monitoring of Hydraulic Systems", "file": filename},
                    )
                metadata = {"source": "UCI Condition Monitoring of Hydraulic Systems", "cycle_number": index}
                if profile_stream:
                    profile = np.fromstring(lines[-1], sep=" ", dtype=np.float64)
                    if profile.size != len(PROFILE_FIELDS):
                        raise ValueError(f"profile.txt cycle {index}: expected 5 values, found {profile.size}")
                    metadata["condition_profile"] = dict(zip(PROFILE_FIELDS, profile.tolist()))
                yield TelemetryCycle(f"hydraulic-{index:04d}", channels, metadata)

