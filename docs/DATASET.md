# UCI Hydraulic Dataset Setup

The project supports the public [UCI Condition Monitoring of Hydraulic Systems](https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems)
dataset. UCI describes 2,205 repeated 60-second hydraulic test-rig cycles and publishes
the archive under CC BY 4.0. The data is external and did not come from an employer.

Download the 73.1 MB archive from the UCI page or directly from:

```text
https://archive.ics.uci.edu/static/public/447/condition+monitoring+of+hydraulic+systems.zip
```

Extract it outside Git. The default configuration expects these files in one directory:

```text
data/hydraulic/PS1.txt
data/hydraulic/FS1.txt
data/hydraulic/TS1.txt
data/hydraulic/VS1.txt
data/hydraulic/EPS1.txt
data/hydraulic/profile.txt
```

Then run:

```bash
telemetry inspect data/hydraulic --config configs/telemetry.yaml
telemetry validate data/hydraulic --config configs/telemetry.yaml
telemetry process data/hydraulic --config configs/telemetry.yaml \
  --backend numba --threads 4 --batch-size 64 --output results/hydraulic
```

The adapter maps UCI names to internal names:

| Internal name | File | Quantity | Rate | Unit |
|---|---|---|---:|---|
| `pressure_1` … `pressure_6` | `PS1.txt` … `PS6.txt` | Pressure | 100 Hz | bar |
| `motor_power` | `EPS1.txt` | Motor power | 100 Hz | W |
| `volume_flow_1`, `volume_flow_2` | `FS1.txt`, `FS2.txt` | Volume flow | 10 Hz | l/min |
| `temperature_1` … `temperature_4` | `TS1.txt` … `TS4.txt` | Temperature | 1 Hz | °C |
| `vibration_1` | `VS1.txt` | Vibration | 1 Hz | mm/s |
| `cooling_efficiency` | `CE.txt` | Cooling efficiency | 1 Hz | % |
| `cooling_power` | `CP.txt` | Cooling power | 1 Hz | kW |
| `efficiency_factor` | `SE.txt` | Efficiency factor | 1 Hz | % |

Each sensor file is a tab/whitespace-delimited matrix: a row is one cycle and columns
are time samples. `profile.txt` contains five cycle-level condition annotations. The
adapter opens selected files together, reads one row from each, verifies equal row
counts and the expected `60 × rate` samples, parses only numeric float64 values, and
yields a `TelemetryCycle`. The labels remain source metadata; the engine does not train
or evaluate a model from them.

