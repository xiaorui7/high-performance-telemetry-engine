"""Confirmed, seeded OOF evaluation against ranking on identical slices."""
import argparse
import json
import os
from pathlib import Path
import tempfile
import shutil
import numpy as np
import pandas as pd
from telemetry_engine.agent.graph import investigate
from telemetry_engine.agent.tools import _load

def choose_slices(size, slices=5, slice_size=100, seed=2026):
    if min(slices,slice_size) < 1 or slices*slice_size > size:
        raise ValueError("not enough cycles for requested nonoverlapping slices")
    # Seeded selection of disjoint contiguous blocks; incomplete trailing block excluded.
    starts=np.random.default_rng(seed).choice(np.arange(size//slice_size)*slice_size,slices,replace=False)
    return [np.arange(start,start+slice_size) for start in sorted(starts)]

def score(called, truth):
    called=set(called)
    if not called <= set(truth.index): raise ValueError("report contains cycle outside slice")
    positives=set(truth.index[truth < 100])
    tp=len(called & positives)
    return dict(degraded_calls=len(called),true_degraded=tp,actual_degraded=len(positives),
                precision=tp/len(called) if called else None,
                recall=tp/len(positives) if positives else None)

def update_docs(report_path, docs_path):
    path=Path(docs_path)
    text=path.read_text(encoding="utf-8")
    marker="<!-- evaluation-results -->"
    text=text.split(marker)[0].rstrip()+"\n\n"+marker+"\n"
    report_path=Path(report_path)
    if not report_path.exists():
        text+="\n## Evaluation results\n\nnot run yet\n"
    else:
        result=json.loads(report_path.read_text(encoding="utf-8"))
        text+="\n## Evaluation results\n\n| Method | Precision | Recall | Grounding pass rate | Avg tool calls | Avg seconds | Total tokens |\n|---|---:|---:|---:|---:|---:|---:|\n"
        for name,row in result["summary"].items():
            values=[row[k] for k in ("precision","recall","grounding_pass_rate","average_tool_calls","average_seconds","total_tokens")]
            text+="| "+name+" | "+" | ".join("N/A" if v is None else f"{v:.4g}" for v in values)+" |\n"
    path.write_text(text,encoding="utf-8")

def evaluate(results, model, table_path, output, slices=5, slice_size=100, seed=2026,
             top_n=3, predictions_source="oof", llm_factory=None):
    if not 1 <= top_n <= 3: raise ValueError("top_n must be between one and three")
    if llm_factory is None and not os.environ.get("TELEMETRY_AGENT_MODEL"):
        raise ValueError("set TELEMETRY_AGENT_MODEL to run real agent evaluation")
    table=pd.read_csv(table_path)
    if table.cycle_id.duplicated().any(): raise ValueError("duplicate labeled cycles")
    table=table.set_index("cycle_id")
    features,predictions=_load(results,model,predictions_source)
    order=table.loc[features.cycle_id].sort_values("cycle_index").index
    features=features.set_index("cycle_id").loc[order].reset_index()
    if not np.all(np.diff(table.loc[order,"cycle_index"]) == 1):
        raise ValueError("evaluation needs consecutive cycle indices")
    selected=choose_slices(len(features),slices,slice_size,seed)
    flags=json.loads((Path(results)/"flags.json").read_text())
    rows=[]
    artifact_root=Path(output).with_suffix("")
    artifact_root.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for index,indices in enumerate(selected):
            subset=features.iloc[indices]; ids=subset.cycle_id.tolist()
            truth=table.loc[ids,"valve_condition_percent"]
            ranked=predictions.set_index("cycle_id").loc[ids].sort_values("p_degraded",ascending=False,kind="stable")
            baseline=score(ranked.head(top_n).index,truth)
            rows.append(dict(method="ranking_baseline",slice=index,**baseline,grounded=None,
                             tool_calls=0,llm_calls=0,seconds=None,total_tokens=0,input_tokens=0,output_tokens=0))
            for version in ("v1","v2"):
                root=Path(tmp)/version/str(index);root.mkdir(parents=True)
                subset.to_csv(root/"features.csv",index=False)
                (root/"flags.json").write_text(json.dumps([r for r in flags if r["cycle_id"] in ids]))
                destination=artifact_root/version/str(index)
                destination.mkdir(parents=True,exist_ok=True)
                try:
                    state=investigate(root,model,table_path,root/"report.md",root/"trace.json",version,
                        input_fn=lambda _: "n", predictions_source=predictions_source,top_n=top_n,
                        llm=llm_factory() if llm_factory else None)
                finally:
                    for filename in ("report.md","trace.json"):
                        if (root/filename).exists():
                            shutil.copyfile(root/filename,destination/filename)
                trace=json.loads((root/"trace.json").read_text())
                usage=[r.get("tokens") for r in trace["trace"] if r["kind"]=="llm"]
                tokens=sum(u["total_tokens"] for u in usage) if usage and all(u and "total_tokens" in u for u in usage) else None
                rows.append(dict(method=version,slice=index,**score(state["degraded_cycles"],truth),
                    grounded=state["grounded"],tool_calls=sum(r["kind"]=="tool" for r in trace["trace"]),
                    seconds=trace["seconds"],total_tokens=tokens,
                    llm_calls=len(usage),
                    input_tokens=sum(u["input_tokens"] for u in usage) if tokens is not None else None,
                    output_tokens=sum(u["output_tokens"] for u in usage) if tokens is not None else None))
    summary={}
    for name in ("v1","v2","ranking_baseline"):
        cases=[r for r in rows if r["method"]==name]
        called=sum(r["degraded_calls"] for r in cases); actual=sum(r["actual_degraded"] for r in cases)
        tp=sum(r["true_degraded"] for r in cases)
        summary[name]=dict(precision=tp/called if called else None,recall=tp/actual if actual else None,
            grounding_pass_rate=float(np.mean([r["grounded"] for r in cases])) if name!="ranking_baseline" else None,
            average_tool_calls=float(np.mean([r["tool_calls"] for r in cases])),
            average_seconds=float(np.mean([r["seconds"] for r in cases])) if name!="ranking_baseline" else None,
            total_tokens=sum(r["total_tokens"] for r in cases) if all(r["total_tokens"] is not None for r in cases) else None)
    result=dict(predictions_source=predictions_source,model=os.environ.get("TELEMETRY_AGENT_MODEL") if llm_factory is None else "scripted-fake",
                seed=seed,top_n=top_n,slice_size=slice_size,
                usage={"llm_calls":sum(r["llm_calls"] for r in rows),
                       **{k:sum(r[k] for r in rows) if all(r[k] is not None for r in rows) else None
                          for k in ("input_tokens","output_tokens","total_tokens")}},
                slices=[features.iloc[idx].cycle_id.tolist() for idx in selected],cases=rows,summary=summary)
    path=Path(output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2),encoding="utf-8")
    return result

def main(argv=None):
    p=argparse.ArgumentParser()
    for name in ("results","model","table"):p.add_argument("--"+name,required=True)
    p.add_argument("--output",default="evals/agent_report.json")
    p.add_argument("--slices",type=int,default=5);p.add_argument("--slice-size",type=int,default=100)
    p.add_argument("--seed",type=int,default=2026);p.add_argument("--top-n",type=int,default=3)
    p.add_argument("--predictions-source",choices=("oof","model"),default="oof")
    p.add_argument("--confirm",action="store_true")
    p.add_argument("--mode",choices=("ranking","uncertain"),default="ranking")
    a=p.parse_args(argv)
    if a.mode == "uncertain":
        planned=2*a.slices*8
        print(f"Planned LLM calls: at most {planned} (two prompts, up to eight candidates per slice).")
        print(f"Estimated list-price cost: under ${planned*(.4*8000+1.6*500)/1e6:.2f} at 8,000 input and 500 output tokens per call.")
    else:
        print(f"Planned LLM calls: at most {2*a.slices*(2*a.top_n+3)} (both prompts, including one report retry per slice).")
    if not a.confirm:p.exit(2,"No calls made. Re-run with --confirm to allow LLM usage.\n")
    if not os.environ.get("TELEMETRY_AGENT_MODEL"):p.exit(2,"Set TELEMETRY_AGENT_MODEL and provider credentials. No calls made.\n")
    if a.mode == "uncertain":
        from telemetry_engine.agent.uncertain import evaluate as evaluate_uncertain, update_docs as update_uncertain_docs
        result=evaluate_uncertain(a.results,a.model,a.table,a.output,a.slices,a.slice_size,a.seed)
        update_uncertain_docs(a.output,Path(__file__).resolve().parents[1]/"docs/AGENT.md")
        if result["status"] == "partial":
            print("Daily request limit reached; partial results saved.")
            return
    else:
        evaluate(a.results,a.model,a.table,a.output,a.slices,a.slice_size,a.seed,a.top_n,a.predictions_source)
    update_docs(a.output,Path(__file__).resolve().parents[1]/"docs/AGENT.md")

if __name__ == "__main__":main()
