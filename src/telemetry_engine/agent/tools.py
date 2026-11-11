"""Deterministic investigation tools; no model or network calls."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from ..ml.predict import predict

def rounded(value):
    if isinstance(value, dict): return {k: rounded(v) for k,v in value.items()}
    if isinstance(value, list): return [rounded(v) for v in value]
    if isinstance(value, (float,np.floating)): return round(float(value),3)
    return value

def _load(results_dir, model_dir, predictions_source="model"):
    try:
        root = Path(results_dir)
        values = pd.read_csv(root / "features.csv")
        if predictions_source == "model":
            predictions = predict(root / "features.csv", model_dir)
        elif predictions_source == "oof":
            predictions = pd.read_csv(Path(model_dir) / "oof_predictions.csv")
            expected = ["cycle_id","cycle_index","fold","p_73","p_80","p_90","p_100","p_degraded"]
            if list(predictions.columns) != expected:
                raise ValueError("invalid OOF schema")
            if predictions.isna().any().any() or predictions.cycle_id.duplicated().any() or predictions.cycle_index.duplicated().any():
                raise ValueError("OOF cycles must be unique and nonmissing")
            probabilities = predictions[expected[3:]].to_numpy(dtype=float)
            if (not np.isfinite(probabilities).all() or (probabilities < 0).any()
                    or (probabilities > 1).any()
                    or not np.allclose(probabilities[:,:4].sum(axis=1),1,atol=1e-6)
                    or not np.allclose(predictions.p_degraded,1-predictions.p_100,atol=1e-6)):
                raise ValueError("invalid OOF probabilities")
            if values.cycle_id.duplicated().any() or values.empty or values.cycle_id.isna().any():
                raise ValueError("input cycles must be nonempty and unique")
            if not set(values.cycle_id) <= set(predictions.cycle_id):
                raise ValueError("OOF predictions missing input cycles")
            predictions = predictions.set_index("cycle_id").loc[values.cycle_id].reset_index()
        else:
            raise ValueError("predictions_source must be model or oof")
        return values, predictions
    except (OSError, KeyError) as exc:
        raise ValueError(f"cannot load investigation input: {exc}") from exc

def triage(results_dir, model_dir, top_n=3, predictions_source="model"):
    if not 1 <= top_n <= 3: raise ValueError("top_n must be between one and three")
    _, predictions = _load(results_dir,model_dir,predictions_source)
    try:
        flags = {r["cycle_id"]:r["flags"] for r in json.loads((Path(results_dir)/"flags.json").read_text())}
    except (OSError,KeyError) as exc: raise ValueError(f"cannot load flags: {exc}") from exc
    predictions["flags"] = predictions.cycle_id.map(lambda c: flags.get(c,[]))
    predictions["flag_count"] = predictions["flags"].map(len)
    return rounded({"cycles": predictions.sort_values(["p_degraded","flag_count"],ascending=False,kind="stable").head(top_n).to_dict("records")})

def compare_to_healthy(results_dir, model_dir, cycle_id, top_k=5):
    try: values = pd.read_csv(Path(results_dir) / "features.csv")
    except OSError as exc: raise ValueError(f"cannot load features: {exc}") from exc
    row = values[values.cycle_id == cycle_id]
    if row.empty: raise ValueError(f"unknown cycle ID: {cycle_id}")
    if top_k < 1: raise ValueError("top_k must be positive")
    try: stats=json.loads((Path(model_dir)/"stats.json").read_text())
    except OSError as exc: raise ValueError(f"missing stats: {exc}") from exc
    features=[]
    for name,mean,std in zip(stats["features"],stats["healthy_mean"],stats["healthy_std"]):
        value=float(row.iloc[0][name])
        z=(value-mean)/std if std > 0 else None
        features.append(dict(feature=name,value=value,healthy_mean=mean,healthy_std=std,z=z,
                             constant_baseline_changed=bool(std == 0 and value != mean)))
    features.sort(key=lambda r:(r["constant_baseline_changed"],abs(r["z"] or 0)),reverse=True)
    return rounded({"cycle_id":cycle_id,"features":features[:top_k]})

def neighbor_trend(results_dir, model_dir, cycle_id, window=5, predictions_source="model"):
    _,p = _load(results_dir,model_dir,predictions_source)
    if window < 1: raise ValueError("window must be positive")
    matches = p.index[p.cycle_id == cycle_id].tolist()
    if not matches: raise ValueError(f"unknown cycle ID: {cycle_id}")
    i=matches[0]; local=p.iloc[max(0,i-window):i+window+1]
    others=local[local.cycle_id != cycle_id]
    persistent=bool(len(others) and (others.p_degraded >= .5).mean() >= .5)
    return rounded({"cycle_id":cycle_id,"trend":"persistent" if persistent else "one-off",
            "neighbors":local[["cycle_id","p_degraded"]].to_dict("records"),
            "assumption":"CSV row order is operating-cycle order; threshold is probability at least 0.5"})

def test_rule(table_csv, feature, threshold, direction):
    try: table=pd.read_csv(table_csv)
    except OSError as exc: raise ValueError(f"cannot read labeled table: {exc}") from exc
    if feature not in table or feature in ("cycle_id","cycle_index","valve_condition_percent"):
        raise ValueError("unknown feature")
    if direction not in (">","<") or not np.isfinite(threshold): raise ValueError("invalid threshold or direction")
    values=table[feature].to_numpy(dtype=float)
    if not np.isfinite(values).all(): raise ValueError("nonfinite rule feature")
    actual=table.valve_condition_percent < 100
    predicted=values > threshold if direction == ">" else values < threshold
    tp=int((predicted & actual).sum())
    return rounded({"feature":feature,"threshold":float(threshold),"direction":direction,
            "precision":tp/int(predicted.sum()) if predicted.sum() else 0.,
            "recall":tp/int(actual.sum()) if actual.sum() else 0.,
            "support":len(table),"evaluation":"descriptive labeled-table evaluation, not held-out"})
