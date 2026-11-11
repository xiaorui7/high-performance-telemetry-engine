"""Offline demonstration; synthetic labels and scripted prose are not evaluation."""
import json
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from ..config import EngineConfig
from ..datasets.synthetic import generate_synthetic_dataset, SyntheticDataset
from ..pipeline import process_to_directory
from ..ml.train import train
from .graph import investigate, build_graph

def scripted_model():
    responses=[json.dumps({"tool":"neighbor_trend"}),json.dumps({"tool":"compare_to_healthy"})]*3
    responses += [json.dumps({"feature":"pressure_1_mean","threshold":10.0,"direction":">"}),
                  "{}"]

    class ScriptedInvestigator(FakeListChatModel):
        def _call(self, *args, **kwargs):
            if self.i < len(self.responses)-1:
                return super()._call(*args, **kwargs)
            messages=args[0]
            evidence=json.loads(messages[-1].content.split("\nEvidence: ",1)[1])
            cycles=evidence[0]["cycles"]
            lines=["# Investigation demo — scripted model output", "",
                   "Synthetic telemetry with artificial labels. These are diagnostic leads, not confirmed faults.",
                   "", "## Triaged cycles", "",
                   "| Cycle | Model p_degraded | Trend | Example tool evidence |",
                   "|---|---:|---|---|"]
            for cycle in cycles:
                cycle_id=cycle["cycle_id"]
                trend=next(item for item in evidence if item.get("cycle_id")==cycle_id and "trend" in item)
                comparison=next(item for item in evidence if item.get("cycle_id")==cycle_id and "features" in item)
                feature=comparison["features"][0]
                lines.append(f"| {cycle_id} | {cycle['p_degraded']} | {trend['trend']} | "
                             f"{feature['feature']} = {feature['value']}; healthy mean = {feature['healthy_mean']} |")
            rule=next(item for item in evidence if "precision" in item and "recall" in item)
            lines += ["", "## Proposed diagnostic rule", "",
                      f"{rule['feature']} {rule['direction']} {rule['threshold']}: "
                      f"measured precision {rule['precision']}, recall {rule['recall']} "
                      "on the synthetic labeled table. Human approval controls saving the proposal.",
                      "", "## Next checks", "",
                      "Review the flagged cycles in the original signals, inspect adjacent cycles, "
                      "and test the rule on independent labeled data before use."]
            return json.dumps({"report":"\n".join(lines),"degraded_cycles":[]})

    return ScriptedInvestigator(responses=responses)

def export_graph(destination):
    graph=build_graph(".",".",".",scripted_model())
    diagram=graph.get_graph().draw_mermaid()
    path=Path(destination);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text("# Compiled investigation graph\n\nGenerated from graph.get_graph().draw_mermaid().\n\n```mermaid\n"+diagram+"\n```\n",encoding="utf-8")
    return diagram

def run_demo(report="report.md",trace="trace.json",yes=False,input_fn=input):
    with tempfile.TemporaryDirectory(prefix="telemetry-demo-") as tmp:
        root=Path(tmp)
        data=generate_synthetic_dataset(root/"data",cycles=40,channels=5,samples=100)
        process_to_directory(SyntheticDataset(data).iter_cycles(),EngineConfig(),root/"results",backend="numpy")
        table=pd.read_csv(root/"results/features.csv")
        table["cycle_index"]=np.arange(len(table))
        table["valve_condition_percent"]=np.tile([73,80,90,100],10)
        table.to_csv(root/"table.csv",index=False)
        train(root/"table.csv",root/"model")
        state=investigate(root/"results",root/"model",root/"table.csv",report,trace,
                          yes=yes,llm=scripted_model(),input_fn=input_fn)
        path=Path(trace)
        payload=json.loads(path.read_text(encoding="utf-8"))
        payload["demo"]=True
        payload["data"]="deterministic synthetic telemetry with artificial cycling labels"
        payload["model"]="scripted-fake"
        path.write_text(json.dumps(payload,indent=2),encoding="utf-8")
        return state
