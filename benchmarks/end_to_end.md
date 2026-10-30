# End-to-End Telemetry Benchmark

This measurement covers cycle-file loading, validation, multi-rate alignment, feature processing, diagnostic rules, and CSV/JSON output. Synthetic dataset generation is outside the timed region.

| Backend | Threads | Cycles | Channels | Raw samples at max rate | Aligned samples | Batch | Warm median | P95 | Throughput | Peak RSS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| numba | 4 | 1000 | 5 | 6000 | 600 | 128 | 2.554201 s | 2.904776 s | 391.5 cycles/s | 136.0 MiB |

The first in-process run took 23.250375 s. The reported steady-state median uses 3 subsequent runs; the first run may include Numba initialization, compilation, or cache loading. Operating-system file caching can benefit later runs. RSS is sampled every 1 ms.

## Environment

- CPU: Intel(R) Core(TM) Ultra 9 285H
- OS: Windows-11-10.0.26200-SP0
- Python: 3.13.12
- NumPy: 2.4.4
- Numba: 0.68.0
