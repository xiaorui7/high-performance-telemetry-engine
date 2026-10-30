# Project Guide

Start with the call path in `telemetry_engine.cli.main()`. A `process` command loads an
`EngineConfig`, opens one dataset adapter, and passes its iterator to
`pipeline.process_to_directory()`. The pipeline validates IDs and channel data, aligns
one batch, selects a function from `features.BACKENDS`, applies `rules.apply_rules()`,
and writes results before requesting the next batch.

The central engineering choice is the reference implementation. Read
`reference_features()` first, then compare `numpy_features()` and `_numba_kernel()`.
They calculate the same eight values but trade readability for lower interpreter
overhead and CPU parallelism. The tests and benchmark convert that shared contract into
evidence rather than assuming faster code is correct.
