"""Join existing pipeline features to public UCI annotations."""
from pathlib import Path
import pandas as pd
from ..config import load_config
from ..datasets.hydraulic import HydraulicDataset, DEFAULT_CHANNELS
from ..pipeline import process_to_directory

def build_table(data_dir, config_path, out_dir):
    config = load_config(config_path)
    dataset = HydraulicDataset(data_dir, config.required_channels or DEFAULT_CHANNELS)
    labels = []
    def cycles():
        for index, cycle in enumerate(dataset.iter_cycles()):
            profile = cycle.metadata.get("condition_profile", {})
            target = profile.get("valve_condition_percent")
            if target not in (100, 90, 80, 73):
                raise ValueError(f"{cycle.cycle_id}: missing or invalid valve label")
            labels.append({"cycle_id": cycle.cycle_id, "cycle_index": index,
                           "valve_condition_percent": int(target)})
            yield cycle
    root = Path(out_dir)
    process_to_directory(cycles(), config, root, backend="numba", input_source="uci-hydraulic")
    features = pd.read_csv(root / "features.csv")
    table = features.merge(pd.DataFrame(labels), on="cycle_id", validate="one_to_one")
    table.to_csv(root / "table.csv", index=False)
    return root / "table.csv"
