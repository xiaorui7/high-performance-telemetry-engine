# Design and Numerical Contract

## Data flow

`open_dataset()` selects a format-specific adapter. Adapters yield `TelemetryCycle`
objects containing named `TelemetryChannel` values, rates, units, and source metadata.
Validation runs before alignment. A stable channel order is established once, and each
batch becomes a contiguous float64 tensor shaped `(cycles, channels, samples)`.

`align_cycles()` uses a common relative-time grid whose endpoint is excluded. It selects
the shortest available sample coverage in the batch. Linear mode uses piecewise linear
interpolation and holds the final observed value during its last sampling interval. Nearest mode maps
each target time to the closest source index. A lower target rate discards high-frequency
content; no anti-alias filter is claimed.

## Feature definitions

For each aligned channel vector `x` of length `N`:

```text
mean = sum(x) / N
std = sqrt(sum((x - mean)^2) / N)       # population standard deviation
rms = sqrt(sum(x^2) / N)
peak_to_peak = max(x) - min(x)
median, p95 = linear-interpolated quantiles at 0.50 and 0.95
```

All implementations also return min and max. The scalar reference uses Python loops
and a sorted list. NumPy uses vectorized reductions. Numba compiles an equivalent loop
and parallelizes the flattened cycle/channel index with `prange`. Each iteration writes
one distinct feature row and uses local accumulators, preventing shared-write races.

Unit tests require `rtol=1e-12` and `atol=1e-12` against the reference. The benchmark
uses a maximum absolute error guard of `1e-10`. These tolerances allow harmless floating
point reduction-order differences; they are not sensor-accuracy claims.

## Bounded memory

Both adapters are iterators. `pipeline.batched()` retains at most `batch_size` source
cycles. The pipeline validates and aligns that list, computes one feature tensor, writes
CSV and JSON output incrementally, then drops references before requesting the next
batch. The global state is limited to cycle IDs for duplicate detection, counters, and
the small channel schema. Very large cycle-ID sets still grow with cycle count; replacing
that exact duplicate check with an external index is a future out-of-core extension.

## Benchmark methodology

Backend cases run in fresh subprocesses with one BLAS/OpenMP thread and explicit Numba
thread counts. The identical seeded aligned tensor is allocated before timing. Each
backend processes it in batches of 128 for three repetitions. Median and p95 wall time,
throughput, warm-up, maximum reference error, and sampled RSS are recorded.

The separate batch study creates and releases one deterministic input batch at a time.
Only feature calls count toward runtime; RSS sampling covers generation and processing.
This isolates the feature engine while measuring batch working-set growth. Numba
compilation is warmed and reported separately. Neither study is end-to-end application
latency.

The end-to-end study generates its synthetic files before timing, then measures the
complete `SyntheticDataset.iter_cycles()` to `process_to_directory()` path. Its timed
scope includes NPZ loading, validation, alignment, feature processing, diagnostic rules,
and all four output artifacts. It reports the first in-process run separately, then uses
three later runs for median and p95. Those later runs may benefit from operating-system
file caching, so the report states that limitation rather than presenting the result as
cold-start latency.
