# High-Performance Telemetry Processing Engine

## About this repository

This is an independent public re-implementation of the telemetry-processing work I did
during my Summer 2026 software engineering internship. The original code and data belong
to the company and remain confidential, so this version was rebuilt after the internship
using the public UCI hydraulic-system dataset and deterministic synthetic data. All
benchmarks and test results in this repository come from this public version.

## Overview

Industrial test equipment produces pressure, flow, temperature, vibration, and power
signals at different sampling rates. This Python tool turns those heterogeneous cycle
files into validated, aligned, feature-level records that engineers can inspect or feed
into downstream systems. It is a local developer tool with explicit data contracts,
bounded-memory batches, interchangeable numerical backends, and reproducible benchmarks.

## What It Does

- streams equipment cycles from a deterministic synthetic format or the public UCI
  hydraulic-system dataset;
- rejects missing channels, malformed files, invalid rates, inconsistent dimensions,
  duplicate cycle IDs, NaN, infinity, and optional out-of-range values;
- aligns multi-rate channels to a configurable common time grid using linear or nearest
  interpolation;
- computes mean, standard deviation, minimum, maximum, RMS, peak-to-peak, median, and
  95th percentile features;
- evaluates configuration-driven diagnostic thresholds without claiming predictive
  maintenance accuracy;
- writes `features.csv`, `flags.json`, `summary.json`, and `run_metadata.json`;
- processes one configurable batch at a time instead of retaining an unbounded workload.

## Architecture

```text
External Dataset (UCI text files or synthetic NPZ cycles)
                              |
                       Dataset Adapter
                              |
                TelemetryCycle / TelemetryChannel
                              |
                    Validation + rate checks
                              |
                Multi-rate alignment/resampling
                              |
        Reference Python / NumPy / parallel Numba backend
                              |
                  Configurable diagnostic rules
                              |
       features.csv + flags.json + summary + run metadata
```

Dataset parsing, business rules, numerical kernels, CLI concerns, and output writing
remain separate. The backend registry gives every implementation the same
`(cycles, channels, samples) -> (cycles, channels, features)` contract.

## Processing Backends

- **Reference Python** uses explicit loops and sorting. It prioritizes readability and
  supplies the numerical oracle.
- **NumPy** performs vectorized reductions and uses bounded batches to prevent a large
  workload-wide temporary allocation.
- **Numba Parallel** compiles the loop structure and uses `prange` across independent
  cycle/channel pairs. Each worker owns its output row, so threads never share writes.

All eight features are computed as float64. Tests require optimized results to agree
with the reference at `rtol=1e-12, atol=1e-12`; the benchmark rejects maximum absolute
error above `1e-10`.

## Dataset

The primary external example is the public
[UCI Condition Monitoring of Hydraulic Systems dataset](https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems)
(Helwig, Pignanelli, and Schütze; DOI `10.24432/C5CW21`; CC BY 4.0). It contains 2,205
60-second cycles. Pressure and motor-power channels are sampled at 100 Hz, flow at
10 Hz, and temperature, vibration, and derived channels at 1 Hz.

The external archive is not committed. Download and extract it so the selected sensor
files and optional `profile.txt` are in one directory:

```text
data/hydraulic/
├── PS1.txt
├── FS1.txt
├── TS1.txt
├── VS1.txt
├── EPS1.txt
└── profile.txt
```

`HydraulicDataset` reads one row from each selected file at a time, validates the
expected 60-second sample count, attaches units and condition-profile metadata, and
converts the row into the internal model. See [docs/DATASET.md](docs/DATASET.md) for all
supported sensors and setup details. Tests, CI, and benchmarks use synthetic data and
do not download UCI files.

## Quick Start

```bash
git clone https://github.com/xiaorui7/high-performance-telemetry-engine.git
cd high-performance-telemetry-engine
python -m venv .venv
```

Activate the environment and install:

```bash
# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

python -m pip install -e ".[dev]"
```

Generate, validate, inspect, and process a small dataset:

```bash
telemetry generate-synthetic data/synthetic --cycles 100 --channels 5 --samples 6000
telemetry validate data/synthetic --config configs/telemetry.yaml
telemetry inspect data/synthetic --config configs/telemetry.yaml
telemetry process data/synthetic --config configs/telemetry.yaml \
  --backend numba --threads 4 --batch-size 64 --output results
telemetry benchmark-e2e --output benchmarks --cycles 1000 --raw-samples 6000 \
  --batch-size 128 --threads 4 --warm-repeats 3
```

## Example Workflow

Input is a directory of individual synthetic cycle archives or the extracted UCI text
files. `process` validates each batch, aligns selected channels, computes features with
one backend, applies rules, writes rows immediately, releases the batch, and continues.

```text
data/synthetic/cycles/cycle_000000.npz
             -> alignment at 10 Hz
             -> pressure_1_mean, vibration_1_rms, ...
             -> HIGH_VIBRATION (only if configured threshold is crossed)
             -> results/features.csv + results/flags.json
```

The generated metadata records backend, actual Numba thread count, batch size,
configuration, channels, input source, package versions, OS, Python version, command,
and UTC timestamp.

## Performance

The saved study used 1,000 cycles × 5 channels × 600 aligned float64 samples on an
Intel Core Ultra 9 285H with Windows 11, Python 3.13.12, NumPy 2.4.4, and Numba 0.68.0.
Each case ran in a fresh process for three repetitions. Times below are medians for
feature processing in batches of 128; input generation, alignment, file I/O, rule
evaluation, and output writing are outside the timed region. Numba compilation is
excluded from steady-state timing and recorded separately.

| Backend | Threads | Workload | Median | Throughput | vs Reference | Max abs error |
|---|---:|---:|---:|---:|---:|---:|
| Reference Python | 1 | 1,000 cycles | 1.254950 s | 796.8 cycles/s | 1.00× | 0 |
| NumPy | 1 | 1,000 cycles | 0.178336 s | 5,607.4 cycles/s | 7.04× | 4.62e-14 |
| Numba | 1 | 1,000 cycles | 0.195116 s | 5,125.1 cycles/s | 6.43× | 2.22e-16 |
| Numba | 2 | 1,000 cycles | 0.074541 s | 13,415.5 cycles/s | 16.84× | 2.22e-16 |
| Numba | 4 | 1,000 cycles | 0.045894 s | 21,789.2 cycles/s | **27.34×** | 2.22e-16 |

These results compare against a deliberately readable Python oracle, not optimized C++
or a distributed system. Full p95, warm-up, environment, and RSS data are in
[`benchmarks/benchmark_report.md`](benchmarks/benchmark_report.md).

### End-to-end performance

A separate system-level benchmark times the complete path from 1,000 on-disk cycle
files through validation, 10 Hz alignment, Numba feature processing, three diagnostic
rules, and CSV/JSON artifact output. Synthetic generation is excluded. After a separately
reported first run, three warm runs had a **2.554 s median**, **2.905 s p95**, and
**391.5 cycles/s** throughput for 5 channels and 600 aligned samples per channel.

| Backend | Threads | Workload | Batch | Median | P95 | Throughput | Sampled peak RSS |
|---|---:|---:|---:|---:|---:|---:|---:|
| Numba | 4 | 1,000 cycles × 5 channels | 128 | 2.554 s | 2.905 s | **391.5 cycles/s** | 136.0 MiB |

The first in-process run took 23.250 s and may include Numba initialization,
compilation, or cache loading. Later runs may benefit from operating-system file
caching. Exact methodology and raw measurements are in
[`benchmarks/end_to_end.md`](benchmarks/end_to_end.md) and
[`benchmarks/end_to_end.json`](benchmarks/end_to_end.json).

## Batch Processing

The batch study generated and released one batch at a time. Feature-call runtime excludes
synthetic generation; sampled process RSS includes both generation and processing.

| Batch | Median feature time | Throughput | Sampled peak RSS | RSS increase |
|---:|---:|---:|---:|---:|
| 32 | 0.048952 s | 20,428.3 cycles/s | 108.8 MiB | 1.7 MiB |
| 64 | 0.036685 s | 27,258.9 cycles/s | 110.4 MiB | 3.9 MiB |
| 128 | 0.054888 s | 18,219.0 cycles/s | 113.1 MiB | 6.4 MiB |
| 256 | 0.048978 s | 20,417.2 cycles/s | 118.9 MiB | 12.1 MiB |

The measurements show the resource trade-off directly: the smallest batch used about
10.4 MiB less peak RSS than batch 256, while batch 64 achieved the highest throughput
in this run. RSS is sampled every millisecond, so brief peaks and allocator reuse can
affect the exact values.

## Correctness

The reference backend is the oracle for the two optimized implementations. Unit tests
also check features against hand-computed values, verify interpolation on known signals,
and exercise malformed data independently of backend agreement. The formal benchmark
found maximum absolute differences of `4.62e-14` for NumPy and `2.22e-16` for Numba,
both below the `1e-10` benchmark guard.

## Testing

```bash
python -m pytest -q
python -m pytest --cov=telemetry_engine --cov-report=term-missing -q
```

The current local result is **41 passed** with **78% measured statement coverage**.
GitHub Actions runs installation, the full suite, CLI help, synthetic generation,
validation, and a Numba processing smoke test on Windows and Linux with Python 3.11
and 3.13.

## Project Structure

```text
src/telemetry_engine/
├── datasets/          UCI and deterministic synthetic adapters
├── model.py           dataset-independent cycle/channel representation
├── validation.py      boundary and cross-cycle validation
├── alignment.py       configurable multi-rate resampling
├── features.py        Reference, NumPy, and Numba feature backends
├── rules.py           threshold-based diagnostic flags
├── pipeline.py        bounded-memory orchestration and outputs
├── benchmark.py       kernel and end-to-end timing, correctness, and RSS studies
└── cli.py             developer commands and error handling
tests/                 unit, integration, failure-path, CLI, and benchmark tests
configs/telemetry.yaml example alignment, batching, validation, and rule configuration
benchmarks/             measured CSV, JSON, and Markdown artifacts
docs/                  dataset setup and design
```

## Engineering Decisions

- **Reference before optimization:** the simple implementation provides an inspectable
  behavioral contract and catches performance changes that would otherwise lack an oracle.
- **Vectorization:** NumPy removes per-sample Python dispatch, but batches bound array
  temporaries and avoid constructing a workload-wide tensor.
- **CPU parallelism:** Numba assigns independent cycle/channel pairs to `prange`
  iterations. Local accumulators and one output owner per iteration avoid data races.
- **Bounded memory:** source adapters yield cycles; the pipeline loads, validates,
  aligns, processes, and writes one batch before releasing it.
- **Multi-rate alignment:** linear interpolation is understandable and deterministic;
  nearest mode is available for discrete-like signals. Downsampling can discard
  high-frequency information, and interpolation does not reconstruct information that
  a low-rate sensor never measured.
- **Correctness before speed:** every optimized result is checked against float64
  reference output, and benchmarks fail when the documented tolerance is exceeded.

## Limitations

- This is a local CPU processing engine, not a distributed or streaming service.
- Diagnostic rules are user-supplied thresholds, not predictive-maintenance models.
- No factory, employer, customer, or production deployment is claimed.
- The UCI dataset is public external demonstration data and is not bundled.
- Linear/nearest resampling is intentionally simple; no anti-alias filter is applied
  before downsampling.
- Feature processing assumes each validated batch can fit in memory; batch size controls
  that bound but does not provide out-of-core arrays within a single cycle.
- RSS sampling is approximate, and benchmark results depend on hardware and system load.
- The project does not infer causal faults or measure diagnostic accuracy.
