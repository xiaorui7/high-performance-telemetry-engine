"""Strict feature-schema inference."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from .train import CLASSES

def predict(features_csv, model_dir):
    root = Path(model_dir)
    columns = json.loads((root / "features.json").read_text())
    table = pd.read_csv(features_csv)
    if list(table.columns) != ["cycle_id"] + columns:
        raise ValueError("feature columns or order do not match the model")
    if table.cycle_id.isna().any() or table.cycle_id.duplicated().any() or table.empty:
        raise ValueError("cycle IDs must be nonmissing and unique; table must not be empty")
    values = table[columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("features must be finite")
    model = XGBClassifier(); model.load_model(root / "model.json")
    probabilities = model.predict_proba(values)
    result = pd.DataFrame({"cycle_id": table.cycle_id,
                           "predicted_class": np.array(CLASSES)[probabilities.argmax(axis=1)]})
    for i,label in enumerate(CLASSES):
        result[f"p_{label}"] = probabilities[:,i]
    result["p_degraded"] = 1 - result["p_100"]
    return result
