"""Candidate-level OOF evaluation with bounded deterministic evidence."""
import json
import os
from pathlib import Path
import re
import time
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from .tools import _load, neighbor_trend, compare_to_healthy, rounded
from .graph import verify_report
from .prompts import PROMPTS

def candidate_ids(predictions, cap=8):
    ranked=predictions.sort_values("p_degraded",ascending=False,kind="stable")
    uncertain=predictions[predictions.p_degraded.between(.3,.7,inclusive="both")]
    # Top three always included; uncertain candidates closest to decision boundary first.
    uncertain=uncertain.assign(distance=(uncertain.p_degraded-.5).abs()).sort_values("distance",kind="stable")
    return list(dict.fromkeys(list(ranked.head(3).cycle_id)+list(uncertain.cycle_id)))[:cap]

def metrics(rows):
    if not rows: return None
    y=[int(row["actual_degraded"]) for row in rows]
    p=[int(row["predicted_degraded"]) for row in rows]
    return dict(precision=float(precision_score(y,p,zero_division=0)),
                recall=float(recall_score(y,p,zero_division=0)),
                f1=float(f1_score(y,p,zero_division=0)),
                accuracy=float(accuracy_score(y,p)),n=len(y))

def context_probability(predictions, cycle_id, window=5):
    index=predictions.index[predictions.cycle_id==cycle_id][0]
    return float(predictions.iloc[max(0,index-window):index+window+1].p_degraded.mean())

def is_daily_limit(error):
    text=str(error).lower()
    return ("rpd" in text or "requests per day" in text or "daily" in text) and ("429" in text or "rate limit" in text)

def invoke_with_retry(llm, messages, attempts=3, sleeper=time.sleep):
    for attempt in range(attempts):
        try: return llm.invoke(messages)
        except Exception as exc:
            if is_daily_limit(exc): raise
            if ("429" not in str(exc) and "rate limit" not in str(exc)) or attempt==attempts-1: raise
            sleeper(2**attempt)
    raise RuntimeError("unreachable")

def evaluate(results, model, table_path, output, slices=3, slice_size=100,
             seed=2026, llm_factory=None):
    from langchain.chat_models import init_chat_model
    from importlib.util import spec_from_file_location, module_from_spec
    script=Path(__file__).resolve().parents[3]/"scripts/eval_agent.py"
    spec=spec_from_file_location("eval_agent",script);mod=module_from_spec(spec);spec.loader.exec_module(mod)
    if llm_factory is None and not os.environ.get("TELEMETRY_AGENT_MODEL"):
        raise ValueError("set TELEMETRY_AGENT_MODEL")
    features,predictions=_load(results,model,"oof")
    labels=pd.read_csv(table_path).set_index("cycle_id")
    order=labels.loc[features.cycle_id].sort_values("cycle_index").index
    predictions=predictions.set_index("cycle_id").loc[order].reset_index()
    indices=mod.choose_slices(len(predictions),slices,slice_size,seed)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    result={"status":"partial","mode":"uncertain","predictions_source":"oof",
            "model":os.environ.get("TELEMETRY_AGENT_MODEL") if llm_factory is None else "scripted-fake",
            "seed":seed,"slices":slices,"slice_size":slice_size,
            "cases":[],"summary":{},"usage":{"llm_calls":0,"input_tokens":0,"output_tokens":0,
            "total_tokens":0},"failures":[]}
    def save():
        grouped={}
        for method in ("baseline_a","baseline_b","v1","v2"):
            rows=[r for r in result["cases"] if r["method"]==method]
            m=metrics([r for r in rows if r.get("predicted_degraded") is not None])
            if m is not None:
                m.update(attempted=len(rows),grounding_pass_rate=(sum(r["grounded"] for r in rows)/len(rows) if method in ("v1","v2") else None),
                    failure_rate=(sum(r["failed"] for r in rows)/len(rows) if method in ("v1","v2") else None),
                    calls=sum(r.get("llm_calls",0) for r in rows),
                    tokens=sum(r.get("total_tokens",0) for r in rows),
                    cost_usd=(sum(r.get("input_tokens",0) for r in rows)*.4+
                              sum(r.get("output_tokens",0) for r in rows)*1.6)/1e6)
                grouped[method]=m
        result["summary"]=grouped
        output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    save()
    for slice_no,idx in enumerate(indices):
        subset=predictions.iloc[idx].reset_index(drop=True)
        ids=candidate_ids(subset)
        result.setdefault("candidate_ids",{})[str(slice_no)]=ids
        for cycle_id in ids:
            actual=bool(labels.loc[cycle_id,"valve_condition_percent"]<100)
            risk=float(subset.set_index("cycle_id").loc[cycle_id,"p_degraded"])
            context=context_probability(subset,cycle_id)
            for method,decision in (("baseline_a",risk>=.5),("baseline_b",context>=.5)):
                result["cases"].append({"method":method,"slice":slice_no,"cycle_id":cycle_id,
                    "actual_degraded":actual,"predicted_degraded":bool(decision),
                    "p_degraded":risk,"context_mean":context})
            save()
        # The two tools are each called once per candidate, within the graph's existing budget.
        # Use full slice context for neighbors, not only the selected candidate list.
        import tempfile
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            original=pd.read_csv(Path(results)/"features.csv").set_index("cycle_id")
            original.loc[subset.cycle_id].reset_index().to_csv(root/"features.csv",index=False)
            for version in ("v1","v2"):
                llm=llm_factory() if llm_factory else init_chat_model(os.environ["TELEMETRY_AGENT_MODEL"],temperature=0)
                for cycle_id in ids:
                    evidence=[rounded(subset.set_index("cycle_id").loc[cycle_id].to_dict())]
                    evidence[0]["cycle_id"]=cycle_id
                    evidence.append(neighbor_trend(root,model,cycle_id,predictions_source="oof"))
                    evidence.append(compare_to_healthy(root,model,cycle_id))
                    prompt=(
                      'For this one cycle return JSON {"cycle_id":"'+cycle_id+
                      '","degraded":true or false,"report":"one evidence-grounded sentence"}. '
                      'Decide valve degraded (<100) using only tool evidence. '
                      'Both neighbor_trend and compare_to_healthy were called for this cycle.')
                    messages=[("system",PROMPTS[version]),("human",prompt+"\nTools: "+json.dumps(evidence))]
                    start=time.perf_counter()
                    try:
                        response=invoke_with_retry(llm,messages)
                        content=json.loads(response.content)
                        if content.get("cycle_id")!=cycle_id or type(content.get("degraded")) is not bool or not isinstance(content.get("report"),str):
                            raise ValueError("invalid decision schema")
                        grounded=verify_report(content["report"],evidence)
                        tokens=response.usage_metadata or {}
                        usage={k:int(tokens.get(k,0)) for k in ("input_tokens","output_tokens","total_tokens")}
                        result["usage"]["llm_calls"]+=1
                        for key in usage:result["usage"][key]+=usage[key]
                        result["cases"].append(dict(method=version,slice=slice_no,cycle_id=cycle_id,
                            actual_degraded=actual,predicted_degraded=content["degraded"],
                            grounded=grounded,failed=not grounded,report=content["report"],
                            llm_calls=1,seconds=time.perf_counter()-start,**usage))
                    except Exception as exc:
                        result["cases"].append(dict(method=version,slice=slice_no,cycle_id=cycle_id,
                            actual_degraded=actual,predicted_degraded=None,grounded=False,
                            failed=True,llm_calls=0,input_tokens=0,output_tokens=0,total_tokens=0,
                            seconds=time.perf_counter()-start))
                        result["failures"].append({"slice":slice_no,"method":version,"cycle_id":cycle_id,
                            "error_type":type(exc).__name__,"daily_limit":is_daily_limit(exc)})
                        save()
                        if is_daily_limit(exc): return result
                        raise
                    save()
    result["status"]="complete";save()
    return result

def update_docs(report_path, docs_path):
    result=json.loads(Path(report_path).read_text(encoding="utf-8"))
    path=Path(docs_path)
    marker="<!-- uncertain-evaluation-results -->"
    text=path.read_text(encoding="utf-8").split(marker)[0].rstrip()+"\n\n"+marker+"\n\n"
    text += "## Uncertain-cycle evaluation\n\n"
    text += f"Status: **{result['status']}**. Source: `{Path(report_path).as_posix()}`. "
    text += "OOF probabilities on public UCI cycles; the health statistics still come from the full training table.\n\n"
    text += "| Method | N | Precision | Recall | F1 | Accuracy | Grounding | Failure | Calls | Tokens | Est. USD |\n"
    text += "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    for name,row in result["summary"].items():
        fields=("n","precision","recall","f1","accuracy","grounding_pass_rate",
                "failure_rate","calls","tokens","cost_usd")
        text+="| "+name+" | "+" | ".join("N/A" if row.get(key) is None else f"{row[key]:.4g}" for key in fields)+" |\n"
    if result["status"] != "complete":
        text+="\nPartial data cannot establish comparative performance. Missing agent decisions are excluded, not imputed.\n"
    path.write_text(text,encoding="utf-8")
