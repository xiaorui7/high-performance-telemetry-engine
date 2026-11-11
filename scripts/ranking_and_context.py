"""Reproduce label-free OOF ranking and context decisions; labels score them only."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from telemetry_engine.agent.uncertain import candidate_ids
from importlib.util import spec_from_file_location, module_from_spec

def scores(truth,prediction):
    y=truth.astype(bool).to_numpy()
    p=np.asarray(prediction,dtype=bool)
    return {"samples":int(len(y)),"degraded":int(y.sum()),
            "predicted_degraded":int(p.sum()),
            "precision":float(precision_score(y,p,zero_division=0)),
            "recall":float(recall_score(y,p,zero_division=0)),
            "f1":float(f1_score(y,p,zero_division=0)),
            "accuracy":float(accuracy_score(y,p))}

def evaluate(oof_path,table_path,output,seed=2026,slices=3,slice_size=100,window=5):
    oof=pd.read_csv(oof_path).sort_values("cycle_index").reset_index(drop=True)
    table=pd.read_csv(table_path)
    if oof.cycle_id.duplicated().any() or table.cycle_id.duplicated().any():
        raise ValueError("duplicate cycle ID")
    if len(oof)!=len(table) or set(oof.cycle_id)!=set(table.cycle_id):
        raise ValueError("OOF and label tables must contain identical cycles")
    labels=table.set_index("cycle_id").loc[oof.cycle_id,"valve_condition_percent"].reset_index(drop=True)
    actual=labels<100
    ranked=oof.assign(actual=actual).sort_values("p_degraded",ascending=False,kind="stable")
    precision_at={str(k):float(ranked.head(k).actual.mean()) for k in (10,20,50,100)}
    spec=spec_from_file_location("eval_agent",Path(__file__).with_name("eval_agent.py"))
    module=module_from_spec(spec);spec.loader.exec_module(module)
    blocks=module.choose_slices(len(oof),slices,slice_size,seed)
    candidate_indices=[]
    for block in blocks:
        subset=oof.iloc[block].reset_index()
        ids=set(candidate_ids(subset))
        candidate_indices.extend(subset.loc[subset.cycle_id.isin(ids),"index"].tolist())
    candidate_indices=sorted(candidate_indices)
    if len(candidate_indices)!=len(set(candidate_indices)):
        raise ValueError("overlapping candidates")
    def compare(frame,truth):
        p=frame.p_degraded.to_numpy()
        means=pd.Series(p).rolling(window*2+1,center=True,min_periods=1).mean().to_numpy()
        return {"baseline_a":scores(truth,p>=.5),
                "baseline_b":scores(truth,means>=.5)}
    # For candidate decisions, neighborhood uses each 100-cycle slice, exactly as in
    # the agent's neighbor_trend tool. Compute means before filtering candidates.
    candidate_a=[];candidate_b=[]
    for block in blocks:
        subset=oof.iloc[block].reset_index()
        ids=set(candidate_ids(subset))
        means=subset.p_degraded.rolling(window*2+1,center=True,min_periods=1).mean()
        selected=subset.cycle_id.isin(ids)
        candidate_a.extend((subset.loc[selected,"p_degraded"]>=.5).tolist())
        candidate_b.extend((means[selected]>=.5).tolist())
    candidate_truth=actual.iloc[candidate_indices]
    result={"source":"blocked-CV OOF XGBoost; public UCI hydraulic labels",
            "cycles":len(oof),"overall_degraded_rate":float(actual.mean()),
            "precision_at":precision_at,
            "candidate_selection":{"seed":seed,"slices":slices,"slice_size":slice_size,
                "candidate_count":len(candidate_indices),"cycle_ids":oof.iloc[candidate_indices].cycle_id.tolist(),
                "rule":"p_degraded in [0.3,0.7] plus top 3 per slice; cap 8"},
            "uncertain_candidates":{"baseline_a":scores(candidate_truth,candidate_a),
                                    "baseline_b":scores(candidate_truth,candidate_b)},
            "all_cycles":compare(oof,actual),
            "neighbor_rule":"mean OOF p_degraded over current cycle ±5 in evaluation scope, threshold 0.5; no labels used in averaging"}
    path=Path(output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--oof",default="models/valve/oof_predictions.csv")
    p.add_argument("--table",default="models/valve/table.csv")
    p.add_argument("--output",default="evals/ranking_and_context.json")
    a=p.parse_args()
    evaluate(a.oof,a.table,a.output)
