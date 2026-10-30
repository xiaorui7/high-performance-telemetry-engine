"""Developer-facing command line interface."""
import argparse
import json
from pathlib import Path

from .benchmark import run_benchmarks, run_end_to_end_benchmark
from .config import load_config
from .datasets.source import open_dataset
from .datasets.synthetic import generate_synthetic_dataset
from .pipeline import process_to_directory
from .validation import validate_cycles


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="telemetry", description="Validate and process heterogeneous industrial telemetry")
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate-synthetic", help="write a deterministic multi-rate dataset")
    generate.add_argument("output", type=Path)
    generate.add_argument("--cycles", type=int, default=100)
    generate.add_argument("--channels", type=int, default=5)
    generate.add_argument("--samples", type=int, default=6000, help="samples at the highest channel rate")
    generate.add_argument("--sampling-rates", nargs="+", type=float)
    generate.add_argument("--seed", type=int, default=2026)
    for name in ("validate", "inspect"):
        command = commands.add_parser(name, help=f"{name} a telemetry dataset")
        command.add_argument("data", type=Path)
        command.add_argument("--config", type=Path)
    process = commands.add_parser("process", help="align, featurize, apply rules, and write artifacts")
    process.add_argument("data", type=Path)
    process.add_argument("--config", type=Path)
    process.add_argument("--backend", choices=("reference", "numpy", "numba"), default="numba")
    process.add_argument("--threads", type=int, default=4)
    process.add_argument("--batch-size", type=int)
    process.add_argument("--output", type=Path, default=Path("results"))
    benchmark = commands.add_parser("benchmark", help="benchmark backends and batch sizes")
    benchmark.add_argument("--output", type=Path, default=Path("benchmarks"))
    benchmark.add_argument("--cycles", type=int, default=1000)
    benchmark.add_argument("--channels", type=int, default=5)
    benchmark.add_argument("--samples", type=int, default=600)
    benchmark.add_argument("--batch-size", type=int, default=128)
    benchmark.add_argument("--batch-sizes", nargs="+", type=int, default=[32, 64, 128, 256])
    benchmark.add_argument("--repeats", type=int, default=3)
    end_to_end = commands.add_parser("benchmark-e2e", help="benchmark the complete file-to-artifacts workflow")
    end_to_end.add_argument("--output", type=Path, default=Path("benchmarks"))
    end_to_end.add_argument("--cycles", type=int, default=1000)
    end_to_end.add_argument("--raw-samples", type=int, default=6000,
                            help="samples at the highest input channel rate")
    end_to_end.add_argument("--batch-size", type=int, default=128)
    end_to_end.add_argument("--threads", type=int, default=4)
    end_to_end.add_argument("--warm-repeats", type=int, default=3)
    end_to_end.add_argument("--seed", type=int, default=2026)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "generate-synthetic":
            path = generate_synthetic_dataset(args.output, args.cycles, args.channels, args.samples,
                                              args.sampling_rates, args.seed)
            print(json.dumps({"dataset": str(path), "cycles": args.cycles, "channels": args.channels}))
            return
        if args.command == "benchmark":
            result = run_benchmarks(args.output, args.cycles, args.channels, args.samples,
                                    args.batch_size, args.repeats, tuple(args.batch_sizes))
            print(json.dumps({"output": str(args.output), "cases": len(result["backends"])}))
            return
        if args.command == "benchmark-e2e":
            result = run_end_to_end_benchmark(args.output, args.cycles, args.raw_samples,
                                              args.batch_size, args.threads,
                                              args.warm_repeats, args.seed)
            print(json.dumps({
                "output": str(args.output),
                "median_runtime_s": result["measurement"]["median_runtime_s"],
                "throughput_cycles_s": result["measurement"]["throughput_cycles_s"],
            }))
            return
        config = load_config(args.config)
        if args.command == "process" and args.batch_size is not None:
            from dataclasses import replace
            config = replace(config, batch_size=args.batch_size)
        dataset = open_dataset(args.data, config.required_channels)
        if args.command == "validate":
            count = validate_cycles(dataset.iter_cycles(), config)
            print(json.dumps({"valid": True, "cycles": count, "source": dataset.source_name}))
        elif args.command == "inspect":
            iterator = dataset.iter_cycles()
            first = next(iterator, None)
            if first is None:
                raise ValueError("input contains no telemetry cycles")
            print(json.dumps({
                "source": dataset.source_name, "first_cycle_id": first.cycle_id,
                "channels": [{"name": channel.name, "sampling_rate_hz": channel.sampling_rate_hz,
                              "unit": channel.unit, "samples": len(channel.values)}
                             for channel in first.channels.values()],
            }, indent=2))
        else:
            summary = process_to_directory(dataset.iter_cycles(), config, args.output, args.backend,
                                           args.threads, f"{dataset.source_name}:{args.data}")
            print(json.dumps(summary, indent=2))
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"telemetry: {exc}\n")


if __name__ == "__main__":
    main()
