"""Dataset format detection kept outside processing business logic."""
from pathlib import Path

from .hydraulic import HydraulicDataset
from .synthetic import SyntheticDataset


def open_dataset(path: str | Path, required_channels: tuple[str, ...] = ()):
    root = Path(path)
    if (root / "manifest.json").is_file():
        return SyntheticDataset(root)
    if (root / "PS1.txt").is_file():
        return HydraulicDataset(root, required_channels or None) if required_channels else HydraulicDataset(root)
    raise ValueError(f"cannot detect dataset format in {root}")
