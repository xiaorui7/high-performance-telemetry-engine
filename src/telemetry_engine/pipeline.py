"""Bounded-memory orchestration from dataset iterator to structured artifacts."""
from collections.abc import Iterable, Iterator
from dataclasses import asdict
import csv
import importlib.metadata
import json
import platform
from pathlib import Path
import sys
import time

import numba

from . import __version__
from .alignment import align_cycles
from .config import EngineConfig
from .features import FEATURE_NAMES, get_backend
from .model import TelemetryCycle
from .rules import apply_rules
from .validation import validate_cycle


def batched(cycles: Iterable[TelemetryCycle], batch_size: int) -> Iterator[list[TelemetryCycle]]:
    batch: list[TelemetryCycle] = []
    for cycle in cycles:
        batch.append(cycle)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _records_for_batch(batch, aligned, features, rules):
    records = []
    for index, cycle_id in enumerate(aligned.cycle_ids):
        values = {
            f"{channel}_{feature}": float(features[index, channel_index, feature_index])
            for channel_index, channel in enumerate(aligned.channel_names)
            for feature_index, feature in enumerate(FEATURE_NAMES)
        }
        records.append({
            "cycle_id": cycle_id,
            "features": values,
            "flags": apply_rules(features[index], aligned.channel_names, rules),
        })
    return records


def process_cycles(cycles: Iterable[TelemetryCycle], config: EngineConfig, backend: str = "numba",
                   threads: int = 4) -> list[dict]:
    if type(threads) is not int or threads < 1:
        raise ValueError("threads must be a positive integer")
    if backend == "numba":
        numba.set_num_threads(min(threads, numba.config.NUMBA_NUM_THREADS))
    function = get_backend(backend)
    seen: set[str] = set()
    channel_names = config.required_channels or None
    records: list[dict] = []
    for batch in batched(cycles, config.batch_size):
        for cycle in batch:
            validate_cycle(cycle, config)
            if cycle.cycle_id in seen:
                raise ValueError(f"duplicate cycle ID: {cycle.cycle_id}")
            seen.add(cycle.cycle_id)
        aligned = align_cycles(batch, config, channel_names)
        channel_names = aligned.channel_names
        records.extend(_records_for_batch(batch, aligned, function(aligned.values), config.rules))
    if not seen:
        raise ValueError("input contains no telemetry cycles")
    return records


def _config_json(config: EngineConfig) -> dict:
    raw = asdict(config)
    raw["required_channels"] = list(config.required_channels)
    raw["value_ranges"] = {key: list(value) for key, value in config.value_ranges.items()}
    return raw


def process_to_directory(cycles: Iterable[TelemetryCycle], config: EngineConfig, output: str | Path,
                         backend: str = "numba", threads: int = 4,
                         input_source: str = "unknown") -> dict:
    if type(threads) is not int or threads < 1:
        raise ValueError("threads must be a positive integer")
    if backend == "numba":
        numba.set_num_threads(min(threads, numba.config.NUMBA_NUM_THREADS))
    function = get_backend(backend)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    features_path = output / "features.csv"
    flags_path = output / "flags.json"
    seen: set[str] = set()
    channel_names = config.required_channels or None
    total_flags = 0
    flagged_cycles = 0
    started = time.perf_counter()
    csv_stream = features_path.open("w", newline="", encoding="utf-8")
    flags_stream = flags_path.open("w", encoding="utf-8")
    writer = None
    first_flag = True
    flags_stream.write("[\n")
    try:
        for batch in batched(cycles, config.batch_size):
            for cycle in batch:
                validate_cycle(cycle, config)
                if cycle.cycle_id in seen:
                    raise ValueError(f"duplicate cycle ID: {cycle.cycle_id}")
                seen.add(cycle.cycle_id)
            aligned = align_cycles(batch, config, channel_names)
            channel_names = aligned.channel_names
            records = _records_for_batch(batch, aligned, function(aligned.values), config.rules)
            if writer is None:
                columns = ["cycle_id"] + [f"{channel}_{feature}" for channel in channel_names for feature in FEATURE_NAMES]
                writer = csv.DictWriter(csv_stream, fieldnames=columns)
                writer.writeheader()
            for record in records:
                writer.writerow({"cycle_id": record["cycle_id"], **record["features"]})
                if not first_flag:
                    flags_stream.write(",\n")
                flags_stream.write(json.dumps({"cycle_id": record["cycle_id"], "flags": record["flags"]}))
                first_flag = False
                total_flags += len(record["flags"])
                flagged_cycles += bool(record["flags"])
        if not seen:
            raise ValueError("input contains no telemetry cycles")
    finally:
        csv_stream.close()
        flags_stream.write("\n]\n")
        flags_stream.close()
    runtime = time.perf_counter() - started
    summary = {
        "cycles_processed": len(seen), "channels_processed": len(channel_names or ()),
        "flagged_cycles": flagged_cycles, "total_flags": total_flags,
        "runtime_seconds": runtime, "throughput_cycles_per_second": len(seen) / runtime,
    }
    metadata = {
        "backend": backend, "thread_count": numba.get_num_threads() if backend == "numba" else 1,
        "batch_size": config.batch_size, "configuration": _config_json(config),
        "number_of_cycles": len(seen), "number_of_channels": len(channel_names or ()),
        "channels": list(channel_names or ()), "input_source": input_source,
        "engine_version": __version__, "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "numba")},
        "command": " ".join(sys.argv),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return summary
