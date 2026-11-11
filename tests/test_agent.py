import json
import pytest
pytest.importorskip("langgraph")
pytest.importorskip("xgboost")
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from test_ml import trained
from telemetry_engine.agent import tools
from telemetry_engine.agent.graph import investigate, verify_report, build_graph
from langgraph.types import Command

def responses():
    return [json.dumps({"tool":"neighbor_trend"}),json.dumps({"tool":"compare_to_healthy"})]*3 + [
        json.dumps({"feature":"pressure_1_mean","threshold":90,"direction":"<"}),
        json.dumps({"report":"Review triage and neighbor_trend evidence; experimental model risk only.","degraded_cycles":[]})]

def test_tools(trained):
    top=tools.triage(trained,trained)
    cycle=top["cycles"][0]["cycle_id"]
    assert len(top["cycles"])==3
    assert tools.compare_to_healthy(trained,trained,cycle)["features"]
    assert tools.neighbor_trend(trained,trained,cycle)["trend"] in ("persistent","one-off")
    assert 0 <= tools.test_rule(trained/"table.csv","pressure_1_mean",90,"<")["precision"] <= 1
    with pytest.raises(ValueError,match="unknown cycle"): tools.neighbor_trend(trained,trained,"missing")
    with pytest.raises(ValueError): tools.triage(trained/"absent",trained)

@pytest.mark.parametrize("approve",[False,True])
def test_graph_roundtrip(trained,approve):
    root=trained/("yes" if approve else "no")
    state=investigate(trained,trained,trained/"table.csv",root/"report.md",root/"trace.json",
                      yes=approve,llm=FakeListChatModel(responses=responses()),input_fn=lambda _: "n")
    assert state["grounded"]
    assert (root/"rules_proposal.yaml").exists() == approve
    trace=json.loads((root/"trace.json").read_text())
    assert sum(r["kind"]=="tool" for r in trace["trace"]) == 8

def test_interrupt_and_resume(trained):
    graph=build_graph(trained,trained,trained/"table.csv",FakeListChatModel(responses=responses()))
    config={"configurable":{"thread_id":"test"}}
    paused=graph.invoke({},config)
    assert paused["__interrupt__"]
    done=graph.invoke(Command(resume=False),config)
    assert done["approved"] is False

def test_verifier_and_retry(trained):
    assert not verify_report("hydraulic-9999 has value 123456.",[{"cycle_id":"hydraulic-0001","value":1.2}])
    script=responses()
    script[-1]=json.dumps({"report":"Invented 987654.","degraded_cycles":[]})
    script.append(json.dumps({"report":"Evidence from triage.","degraded_cycles":[]}))
    state=investigate(trained,trained,trained/"table.csv",trained/"r.md",trained/"t.json",
                      llm=FakeListChatModel(responses=script),input_fn=lambda _: "n")
    assert state["attempts"]==2 and state["grounded"]


def test_persistent_grounding_failure(trained):
    script=responses()
    script[-1]=json.dumps({"report":"Invented 987654.","degraded_cycles":[]})
    script.append(script[-1])
    state=investigate(trained,trained,trained/"table.csv",trained/"bad.md",trained/"bad-trace.json",
                      llm=FakeListChatModel(responses=script),input_fn=lambda _: "n")
    assert state["attempts"]==2 and not state["grounded"]
    assert (trained/"bad.md").read_text().startswith("WARNING:")


def test_oof_tools_never_predict(trained,monkeypatch):
    import pandas as pd
    def forbidden(*args,**kwargs): raise AssertionError("final model used")
    monkeypatch.setattr(tools,"predict",forbidden)
    top=tools.triage(trained,trained,predictions_source="oof")["cycles"]
    oof=pd.read_csv(trained/"oof_predictions.csv").set_index("cycle_id")
    for cycle in top:
        assert cycle["p_degraded"]==pytest.approx(round(float(oof.loc[cycle["cycle_id"],"p_degraded"]),3))
    trend=tools.neighbor_trend(trained,trained,top[0]["cycle_id"],predictions_source="oof")
    assert all(r["p_degraded"]==pytest.approx(round(float(oof.loc[r["cycle_id"],"p_degraded"]),3)) for r in trend["neighbors"])
    oof.iloc[1:].reset_index().to_csv(trained/"oof_predictions.csv",index=False)
    with pytest.raises(ValueError,match="missing"):tools.triage(trained,trained,predictions_source="oof")

def test_eval_fake_and_baseline(trained,tmp_path,monkeypatch):
    import importlib.util
    from pathlib import Path
    spec=importlib.util.spec_from_file_location("eval_agent",Path(__file__).resolve().parents[1]/"scripts/eval_agent.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.delenv("TELEMETRY_AGENT_MODEL",raising=False)
    def forbidden(*args,**kwargs):raise AssertionError("final model used")
    monkeypatch.setattr(tools,"predict",forbidden)
    result=module.evaluate(trained,trained,trained/"table.csv",tmp_path/"evaluation.json",
                          slices=2,slice_size=10,llm_factory=lambda:FakeListChatModel(responses=responses()))
    assert set(result["summary"])=={"v1","v2","ranking_baseline"}
    assert result["predictions_source"]=="oof"
    for row in result["summary"].values():
        assert set(row)=={"precision","recall","grounding_pass_rate","average_tool_calls","average_seconds","total_tokens"}
    assert result["summary"]["v1"]["recall"]==0
    assert result["summary"]["v1"]["total_tokens"] is None
    for row in result["cases"]:
        if row["method"]=="ranking_baseline":
            assert row["degraded_calls"]==3
            assert row["recall"]==row["true_degraded"]/row["actual_degraded"]
    first=module.choose_slices(40,2,10,7)
    second=module.choose_slices(40,2,10,7)
    assert [x.tolist() for x in first]==[x.tolist() for x in second]
    assert len(set(i for block in first for i in block))==20
    docs=tmp_path/"AGENT.md";docs.write_text("Guide")
    module.update_docs(tmp_path/"evaluation.json",docs)
    assert "ranking_baseline" in docs.read_text()
    with pytest.raises(SystemExit) as error:
        module.main(["--results","x","--model","x","--table","x"])
    assert error.value.code==2

def test_offline_demo(tmp_path):
    from telemetry_engine.agent.demo import run_demo,export_graph
    state=run_demo(tmp_path/"report.md",tmp_path/"trace.json",yes=True)
    assert state["grounded"] and state["approved"]
    trace=json.loads((tmp_path/"trace.json").read_text())
    assert trace["demo"] and trace["model"]=="scripted-fake"
    report=(tmp_path/"report.md").read_text(encoding="utf-8")
    assert "scripted model output" in report
    assert "Triaged cycles" in report and "Proposed diagnostic rule" in report
    assert "precision" in report and "recall" in report and "Next checks" in report
    diagram=export_graph(tmp_path/"graph.md")
    assert "human_approval" in diagram and "verify" in diagram

def test_invalid_proposal_is_rejected(trained):
    script=responses()
    script[-2]=json.dumps({"feature":"invented","threshold":2,"direction":">"})
    state=investigate(trained,trained,trained/"table.csv",trained/"rejected.md",trained/"rejected.json",
                      llm=FakeListChatModel(responses=script),input_fn=lambda _: "n")
    assert not state["approved"] and not state["proposal"]
    trace=json.loads((trained/"rejected.json").read_text())
    assert any(row.get("name")=="rule_rejected" for row in trace["trace"])

def test_rounded_grounding():
    assert verify_report("hydraulic-0001: 0.33 and 1.235",{"cycle_id":"hydraulic-0001","values":[0.333,1.235]})
    assert not verify_report("hydraulic-0002: 0.33",{"cycle_id":"hydraulic-0001","value":0.333})
    assert not verify_report("hydraulic-0001: 0.36",{"cycle_id":"hydraulic-0001","value":0.333})

def test_uncertain_candidates_context_and_partial(trained,tmp_path):
    import pandas as pd
    from telemetry_engine.agent.uncertain import candidate_ids, context_probability, evaluate, update_docs
    frame=pd.DataFrame({"cycle_id":["a","b","c","d","e"],
                        "p_degraded":[.99,.8,.6,.4,.1]})
    assert candidate_ids(frame)==["a","b","c","d"]
    assert context_probability(frame,"c",window=1)==pytest.approx(.6)
    class DailyLimit:
        def invoke(self,messages):
            raise RuntimeError("429 requests per day RPD")
    result=evaluate(trained,trained,trained/"table.csv",tmp_path/"partial.json",
                    slices=1,slice_size=10,llm_factory=lambda:DailyLimit())
    assert result["status"]=="partial"
    assert result["summary"]["baseline_a"]["n"]==len(result["candidate_ids"]["0"])
    assert "v1" not in result["summary"]
    docs=tmp_path/"AGENT.md";docs.write_text("Guide",encoding="utf-8")
    update_docs(tmp_path/"partial.json",docs)
    assert "Partial data" in docs.read_text(encoding="utf-8")


def test_ranking_context_uses_probabilities_not_labels(tmp_path):
    import importlib.util
    import pandas as pd
    from pathlib import Path
    script=Path(__file__).resolve().parents[1]/"scripts/ranking_and_context.py"
    spec=importlib.util.spec_from_file_location("ranking_context",script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    ids=[f"hydraulic-{i:04d}" for i in range(1,21)]
    oof=pd.DataFrame({"cycle_id":ids,"cycle_index":range(20),
                      "p_degraded":[i/20 for i in range(20)]})
    labels=pd.DataFrame({"cycle_id":ids,"valve_condition_percent":[100]*10+[80]*10})
    oof.to_csv(tmp_path/"oof.csv",index=False)
    labels.to_csv(tmp_path/"table.csv",index=False)
    first=module.evaluate(tmp_path/"oof.csv",tmp_path/"table.csv",tmp_path/"out.json",
                          slices=1,slice_size=10)
    labels["valve_condition_percent"]=[80]*10+[100]*10
    labels.to_csv(tmp_path/"table.csv",index=False)
    second=module.evaluate(tmp_path/"oof.csv",tmp_path/"table.csv",tmp_path/"out.json",
                           slices=1,slice_size=10)
    assert first["all_cycles"]["baseline_b"]["predicted_degraded"]==second["all_cycles"]["baseline_b"]["predicted_degraded"]
    assert first["candidate_selection"]["cycle_ids"]==second["candidate_selection"]["cycle_ids"]
    assert second["precision_at"]["10"]==0
