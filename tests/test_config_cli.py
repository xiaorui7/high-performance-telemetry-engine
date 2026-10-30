import json
import pytest

from telemetry_engine.cli import main
from telemetry_engine.config import load_config


def test_config_and_rules(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("target_rate_hz: 5\nbatch_size: 2\nrules:\n  sensor_rms: {max: 4, flag: HIGH}\n")
    config = load_config(path)
    assert config.target_rate_hz == 5 and config.rules["sensor_rms"].flag == "HIGH"


@pytest.mark.parametrize("text", ["[]", "unknown: 1", "target_rate_hz: 0", "alignment: cubic",
                                  "rules: {x: {}}", "value_ranges: {x: [2, 1]}"])
def test_invalid_config(tmp_path, text):
    path = tmp_path / "bad.yaml"
    path.write_text(text)
    with pytest.raises(ValueError):
        load_config(path)


def test_cli_end_to_end(tmp_path, capsys):
    data = tmp_path / "data"
    main(["generate-synthetic", str(data), "--cycles", "3", "--channels", "2", "--samples", "40",
          "--sampling-rates", "4", "2"])
    main(["inspect", str(data)])
    main(["validate", str(data)])
    output = tmp_path / "output"
    main(["process", str(data), "--backend", "numpy", "--batch-size", "2", "--output", str(output)])
    assert json.loads((output / "summary.json").read_text())["cycles_processed"] == 3
    benchmark = tmp_path / "benchmark"
    main(["benchmark-e2e", "--output", str(benchmark), "--cycles", "2",
          "--raw-samples", "20", "--batch-size", "1", "--threads", "1",
          "--warm-repeats", "1"])
    assert json.loads((benchmark / "end_to_end.json").read_text())["workload"]["cycles"] == 2
    assert "synthetic" in capsys.readouterr().out


def test_cli_bad_input(tmp_path):
    with pytest.raises(SystemExit) as error:
        main(["validate", str(tmp_path)])
    assert error.value.code == 2
