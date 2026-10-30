# Telemetry Engine Benchmark

Steady-state timings cover aligned feature processing in bounded batches. Numba warm-up is excluded and reported separately.

| Backend | Threads | Cycles | Channels × Samples | Batch | Median s | P95 s | Cycles/s | vs Reference | vs Numba 1T | Max abs error | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| reference | 1 | 1000 | 5 × 600 | 128 | 1.254950 | 1.369288 | 796.8 | 1.00× | 0.16× | 0.00e+00 | 92.9 |
| numpy | 1 | 1000 | 5 × 600 | 128 | 0.178336 | 0.184296 | 5607.4 | 7.04× | 1.09× | 4.62e-14 | 97.8 |
| numba | 1 | 1000 | 5 × 600 | 128 | 0.195116 | 0.203885 | 5125.1 | 6.43× | 1.00× | 2.22e-16 | 130.5 |
| numba | 2 | 1000 | 5 × 600 | 128 | 0.074541 | 0.075598 | 13415.5 | 16.84× | 2.62× | 2.22e-16 | 130.4 |
| numba | 4 | 1000 | 5 × 600 | 128 | 0.045894 | 0.047795 | 21789.2 | 27.34× | 4.25× | 2.22e-16 | 130.6 |

## Batch-size study (Numba, 4 threads)

| Batch size | Median s | Cycles/s | Peak RSS MiB | RSS increase MiB |
|---:|---:|---:|---:|---:|
| 32 | 0.048952 | 20428.3 | 108.8 | 1.7 |
| 64 | 0.036685 | 27258.9 | 110.4 | 3.9 |
| 128 | 0.054888 | 18219.0 | 113.1 | 6.4 |
| 256 | 0.048978 | 20417.2 | 118.9 | 12.1 |

The batch study generates and releases one input batch at a time. Runtime includes feature calls only; sampled RSS includes batch generation and processing. Memory is sampled every 1 ms, so short peaks and allocator reuse can affect it.
