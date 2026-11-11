"""Bounded LangGraph investigation with resumable human approval."""
import json
import os
from pathlib import Path
import re
import time
from typing import TypedDict
from uuid import uuid4
import yaml
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command
from . import tools
from .prompts import PROMPTS

class State(TypedDict, total=False):
    evidence: list
    cycles: list
    proposal: dict
    approved: bool
    report: str
    degraded_cycles: list
    attempts: int
    grounded: bool
    warning: str

NUMBER = r"(?<![A-Za-z_])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"

def verify_report(report, evidence):
    from decimal import Decimal, ROUND_HALF_EVEN
    source=json.dumps(evidence)
    identifiers=r"hydraulic-\d+|synthetic-\d+"
    ids=set(re.findall(identifiers,report))
    known=set(re.findall(identifiers,source))
    if not ids <= known: return False
    # IDs are labels, not measured numeric values.
    clean_report=re.sub(identifiers,"",report)
    values=[Decimal(token) for token in re.findall(NUMBER,re.sub(identifiers,"",source))]
    for token in re.findall(NUMBER,clean_report):
        candidate=Decimal(token)
        quantum=Decimal(1).scaleb(candidate.as_tuple().exponent)
        if not any(v.quantize(quantum,rounding=ROUND_HALF_EVEN)==candidate for v in values):
            return False
    return True

def build_graph(results_dir, model_dir, table_csv, llm, prompt_version="v2", trace=None, predictions_source="model", top_n=3):
    log = trace if trace is not None else []
    def call_tool(name, **kwargs):
        start=time.perf_counter()
        output=getattr(tools,name)(**kwargs)
        log.append(dict(kind="tool",name=name,inputs={k:str(v) if isinstance(v,Path) else v for k,v in kwargs.items()},
                        output=output,seconds=time.perf_counter()-start))
        return output
    def ask(request, evidence):
        messages=[("system",PROMPTS[prompt_version]),("human",request+"\nEvidence: "+json.dumps(evidence))]
        start=time.perf_counter(); response=llm.invoke(messages)
        log.append(dict(kind="llm",inputs=messages,output=response.content,
                        tokens=getattr(response,"usage_metadata",None),seconds=time.perf_counter()-start))
        try: value=json.loads(response.content)
        except (TypeError,json.JSONDecodeError) as exc: raise ValueError("LLM must return valid JSON") from exc
        if not isinstance(value,dict): raise ValueError("LLM response must be an object")
        return value
    def triage(state):
        output=call_tool("triage",results_dir=results_dir,model_dir=model_dir,
                         predictions_source=predictions_source,top_n=top_n)
        return {"cycles":output["cycles"],"evidence":[output],"attempts":0}
    def investigate_node(state):
        evidence=list(state["evidence"])
        for cycle in state["cycles"]:
            used=set()
            for _ in range(2):
                request = ('Choose next tool for '+cycle["cycle_id"]+
                           '. Already used tools: '+str(sorted(used))+'. Do not repeat them. Return {"tool":"neighbor_trend"} or {"tool":"compare_to_healthy"} or {"tool":"done"}.')
                choice=ask(request,evidence).get("tool")
                if choice == "done": break
                if choice not in ("neighbor_trend","compare_to_healthy") or choice in used:
                    raise ValueError("invalid or duplicate investigation tool")
                used.add(choice)
                evidence.append(call_tool(choice,results_dir=results_dir,model_dir=model_dir,cycle_id=cycle["cycle_id"],
                    **({"predictions_source":predictions_source} if choice == "neighbor_trend" else {})))
        return {"evidence":evidence}
    def propose(state):
        proposal=ask('Suggest optional rule. Return {"feature":null} to skip, or {"feature":"name","threshold":0.0,"direction":">"}.',state["evidence"])
        evidence=list(state["evidence"])
        if proposal.get("feature") is not None:
            try:
                proposal=call_tool("test_rule",table_csv=table_csv,feature=proposal["feature"],
                                   threshold=proposal["threshold"],direction=proposal["direction"])
                evidence.append(proposal)
            except (ValueError,KeyError,TypeError):
                rejection={"rule_status":"rejected","reason":"invalid proposed feature or threshold"}
                evidence.append(rejection)
                log.append(dict(kind="validation",name="rule_rejected",output=rejection))
                proposal={}
        else: proposal={}
        return {"proposal":proposal,"evidence":evidence}
    def approval(state):
        if not state["proposal"]: return {"approved":False}
        answer=interrupt({"proposal":state["proposal"],"question":"Save experimental proposal? y/n"})
        return {"approved":answer is True}
    def report(state):
        value=ask('Write short Markdown report. Return {"report":"text","degraded_cycles":["cycle ID"]}. '
                  'degraded_cycles lists only cycles explicitly called degraded in report. Cite tool names. '
                  'Only copy exact numbers from evidence; no numbered lists. Previous grounding failure: '+
                  str(state.get("warning","")),state["evidence"])
        text=value.get("report")
        if not isinstance(text,str): raise ValueError("report must be text")
        return {"report":text,"degraded_cycles":value.get("degraded_cycles",[]),
                "attempts":state["attempts"]+1}
    def verify(state):
        known={c["cycle_id"] for c in state["cycles"]}
        grounded=verify_report(state["report"],state["evidence"]) and set(state["degraded_cycles"]) <= known
        if prompt_version == "v2":
            checked={e["cycle_id"] for e in state["evidence"] if "trend" in e}
            grounded=grounded and set(state["degraded_cycles"]) <= checked
        return {"grounded":grounded,"warning":"" if grounded else "WARNING: report failed deterministic grounding verification."}
    graph=StateGraph(State)
    for name,fn in [("triage",triage),("investigate",investigate_node),("propose_rule",propose),
                    ("human_approval",approval),("write_report",report),("verify",verify)]:
        graph.add_node(name,fn)
    graph.add_edge(START,"triage")
    for a,b in zip(["triage","investigate","propose_rule","human_approval"],["investigate","propose_rule","human_approval","write_report"]):
        graph.add_edge(a,b)
    graph.add_edge("write_report","verify")
    graph.add_conditional_edges("verify",lambda s: "write_report" if not s["grounded"] and s["attempts"] < 2 else END,
                                {"write_report":"write_report", END:END})
    return graph.compile(checkpointer=InMemorySaver())

def investigate(results_dir, model_dir, table_csv, report_path, trace_path="trace.json",
                prompt_version="v2", yes=False, llm=None, input_fn=input, predictions_source="model", top_n=3):
    report_path = Path(report_path)
    if "configs" in report_path.resolve().parts:
        raise ValueError("investigation outputs must be outside configs/")
    if llm is None:
        from langchain.chat_models import init_chat_model
        model=os.environ.get("TELEMETRY_AGENT_MODEL")
        if not model: raise ValueError("set TELEMETRY_AGENT_MODEL and install its provider package")
        llm=init_chat_model(model,temperature=0)
    trace=[]; started=time.perf_counter()
    graph=build_graph(results_dir,model_dir,table_csv,llm,prompt_version,trace,predictions_source,top_n)
    config={"configurable":{"thread_id":str(uuid4())}}
    try:
        state=graph.invoke({},config)
    except Exception as exc:
        failure_path=Path(trace_path)
        failure_path.parent.mkdir(parents=True,exist_ok=True)
        failure_path.write_text(json.dumps({"status":"failed","error_type":type(exc).__name__,
            "prompt_version":prompt_version,"trace":trace,"seconds":time.perf_counter()-started},indent=2),encoding="utf-8")
        raise
    if state.get("__interrupt__"):
        print(json.dumps(state["__interrupt__"][0].value,indent=2))
        approved=yes or input_fn("Save proposal? [y/N] ").strip().lower() == "y"
        state=graph.invoke(Command(resume=approved),config)
    report_path=Path(report_path); report_path.parent.mkdir(parents=True,exist_ok=True)
    if state["approved"]:
        proposal=state["proposal"]
        artifact=report_path.parent/"rules_proposal.yaml"
        with artifact.open("x",encoding="utf-8") as f:
            yaml.safe_dump({"rules":{proposal["feature"]:{"max" if proposal["direction"] == ">" else "min":proposal["threshold"]}}},f)
    report_path.write_text((state["warning"]+"\n" if state["warning"] else "")+state["report"],encoding="utf-8")
    trace_path=Path(trace_path); trace_path.parent.mkdir(parents=True,exist_ok=True)
    trace_path.write_text(json.dumps({"prompt_version":prompt_version,"predictions_source":predictions_source,"trace":trace,"grounded":state["grounded"],
                                     "degraded_cycles":state["degraded_cycles"],"approved":state["approved"],
                                     "proposal":state["proposal"],"seconds":time.perf_counter()-started},indent=2),encoding="utf-8")
    return state
