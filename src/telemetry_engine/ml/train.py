"""Fixed-seed evaluation; preprocessing is fitted inside each fold."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

CLASSES = [73, 80, 90, 100]

def blocked_folds(size):
    return list(KFold(5, shuffle=False).split(np.arange(size)))

def train(table_csv, out_dir, seed=2026):
    table = pd.read_csv(table_csv).sort_values("cycle_index")
    if (table.cycle_id.isna().any() or table.cycle_id.duplicated().any()
            or table.cycle_index.isna().any() or table.cycle_index.duplicated().any()
            or not np.isfinite(table.cycle_index.to_numpy(dtype=float)).all()):
        raise ValueError("cycle IDs and indices must be nonmissing and unique")
    columns = [c for c in table if c not in ("cycle_id", "cycle_index", "valve_condition_percent")]
    x = table[columns].to_numpy(dtype=float)
    if not columns or not np.isfinite(x).all():
        raise ValueError("features must be finite numeric columns")
    if set(table.valve_condition_percent) != set(CLASSES):
        raise ValueError("training requires all four valve classes")
    y = np.array([CLASSES.index(v) for v in table.valve_condition_percent])
    if min(np.bincount(y)) < 5:
        raise ValueError("each valve class needs at least five cycles")
    factories = {
        "majority": lambda: DummyClassifier(strategy="most_frequent"),
        "logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, random_state=seed)),
        "xgboost": lambda: XGBClassifier(random_state=seed, n_jobs=1, n_estimators=100, max_depth=6),
    }
    splits = {"blocked": blocked_folds(len(y)),
              "stratified": list(StratifiedKFold(5, shuffle=True, random_state=seed).split(x,y))}
    metrics = {"seed": seed, "cycles": len(y), "classes": CLASSES, "results": {}}
    matrices = []
    oof_probabilities = np.empty((len(y),len(CLASSES)))
    oof_folds = np.empty(len(y),dtype=int)
    for split, folds in splits.items():
        metrics["results"][split] = {}
        for name, factory in factories.items():
            prediction = np.empty_like(y)
            for fold, (train_ids, test_ids) in enumerate(folds):
                if name == "xgboost" and set(y[train_ids]) != set(range(4)):
                    raise ValueError("blocked training fold lacks a class; cannot fit fixed-class XGBoost")
                model = factory()
                model.fit(x[train_ids], y[train_ids])
                prediction[test_ids] = model.predict(x[test_ids])
                if split == 'blocked' and name == 'xgboost':
                    oof_probabilities[test_ids] = model.predict_proba(x[test_ids])
                    oof_folds[test_ids] = fold
            cm = confusion_matrix(y, prediction, labels=range(4))
            metrics["results"][split][name] = {
                "macro_f1": float(f1_score(y,prediction,average="macro")),
                "accuracy": float(accuracy_score(y,prediction)),
                "confusion_matrix": cm.tolist()}
            for i,a in enumerate(CLASSES):
                for j,b in enumerate(CLASSES):
                    matrices.append(dict(split=split, model=name, actual=a, predicted=b, count=int(cm[i,j])))
    root = Path(out_dir); root.mkdir(parents=True,exist_ok=True)
    oof = table[["cycle_id","cycle_index"]].reset_index(drop=True).copy()
    oof["fold"] = oof_folds
    for index,label in enumerate(CLASSES):
        oof[f"p_{label}"] = oof_probabilities[:,index]
    oof["p_degraded"] = 1 - oof["p_100"]
    oof.to_csv(root / "oof_predictions.csv",index=False)
    final = factories["xgboost"](); final.fit(x,y); final.save_model(root / "model.json")
    def save(name, value):
        (root / name).write_text(json.dumps(value,indent=2)+"\n",encoding="utf-8")
    save("features.json",columns)
    healthy = x[y == CLASSES.index(100)]
    save("stats.json", {"classes": CLASSES, "features": columns,
         "mean": x.mean(axis=0).tolist(), "std": x.std(axis=0).tolist(),
         "healthy_mean": healthy.mean(axis=0).tolist(), "healthy_std": healthy.std(axis=0).tolist()})
    gain = final.get_booster().get_score(importance_type="gain")
    importance = sorted([{"feature": c, "gain": float(gain.get(f"f{i}",0))} for i,c in enumerate(columns)],
                        key=lambda r:r["gain"],reverse=True)[:20]
    metrics["top_features"] = importance
    save("metrics.json",metrics)
    pd.DataFrame(matrices).to_csv(root / "confusion_matrix.csv",index=False)
    pd.DataFrame(importance).to_csv(root / "feature_importance.csv",index=False)
    write_model_card(root / "metrics.json",root / "MODEL_CARD.md")
    return metrics

def write_model_card(metrics_path, destination):
    m = json.loads(Path(metrics_path).read_text())
    lines = ["# Experimental valve condition model", "",
      "Target: UCI valve condition classes " + ", ".join(map(str,m["classes"])) + ".",
      "Public hydraulic lab-rig telemetry, default five channels; no employer data.",
      "Blocked contiguous cross-validation is primary. Shuffled stratification is a comparison:",
      "adjacent cycles can be correlated. Blocked CV is not forward-only forecasting.",
      "Preprocessing is fitted inside each training fold; metrics use pooled out-of-fold predictions.",
      "", "| Split | Model | Macro-F1 | Accuracy |", "|---|---|---:|---:|"]
    for split, models in m["results"].items():
        for name, scores in models.items():
            lines.append(f"| {split} | {name} | {scores['macro_f1']:.4f} | {scores['accuracy']:.4f} |")
    lines += ["", "## Top gain features", ""] + [f"- {r['feature']}: {r['gain']:.6g}" for r in m["top_features"][:10]]
    lines += ["", "## Limitations", "One public lab rig; five channels only; no drift monitoring; not validated on other equipment.",
              "Final model fitted to all data; its predictions are not held-out evaluation.",
              "stats.json contains all-training and healthy-training statistics; neither is cross-validation evidence.",
              "", "## Next steps for the team",
              "Other targets (cooler, pump, accumulator); all-sensor ablation; random forest;",
              "permutation importance; hyperparameter tuning; drift monitoring; retraining schedule."]
    Path(destination).write_text("\n".join(lines)+"\n",encoding="utf-8")
