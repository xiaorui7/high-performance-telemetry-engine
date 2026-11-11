import json
import numpy as np
import pytest
pytest.importorskip("sklearn")
pytest.importorskip("xgboost")
import pandas as pd
from telemetry_engine.ml.train import train, blocked_folds
from telemetry_engine.ml.predict import predict

@pytest.fixture
def trained(tmp_path):
    rng=np.random.default_rng(7)
    y=np.tile([73,80,90,100],10)
    table=pd.DataFrame({"cycle_id":[f"hydraulic-{i:04d}" for i in range(40)],
        "pressure_1_mean":y+rng.normal(0,.1,40),"temperature_1_max":rng.normal(size=40),
        "cycle_index":np.arange(40),"valve_condition_percent":y})
    path=tmp_path/"table.csv"; table.to_csv(path,index=False)
    train(path,tmp_path)
    table.drop(columns=["cycle_index","valve_condition_percent"]).to_csv(tmp_path/"features.csv",index=False)
    (tmp_path/"flags.json").write_text(json.dumps([{"cycle_id":c,"flags":[]} for c in table.cycle_id]))
    return tmp_path

def test_roundtrip_and_schema(trained):
    result=predict(trained/"features.csv",trained)
    assert len(result)==40
    assert np.allclose(result[["p_73","p_80","p_90","p_100"]].sum(axis=1),1)
    assert np.allclose(result.p_degraded,1-result.p_100)
    table=pd.read_csv(trained/"features.csv").rename(columns={"pressure_1_mean":"wrong"})
    table.to_csv(trained/"bad.csv",index=False)
    with pytest.raises(ValueError,match="columns"): predict(trained/"bad.csv",trained)

def test_blocked_folds():
    observed=[]
    for train_ids,test_ids in blocked_folds(40):
        assert not set(train_ids)&set(test_ids)
        assert np.all(np.diff(test_ids)==1)
        observed.extend(test_ids)
    assert sorted(observed)==list(range(40))

def test_reproducible_metrics(trained):
    before=json.loads((trained/"metrics.json").read_text())
    after=train(trained/"table.csv",trained)
    assert before == after
    assert set(after["results"])=={"blocked","stratified"}
    assert set(after["results"]["blocked"])=={"majority","logistic","xgboost"}
    assert np.array(after["results"]["blocked"]["xgboost"]["confusion_matrix"]).sum()==40


def test_adapter_join(tmp_path):
    from telemetry_engine.ml.dataset import build_table
    data=tmp_path/"data";data.mkdir()
    for name, count in [("PS1",6000),("FS1",600),("TS1",60),("VS1",60),("EPS1",6000)]:
        (data/f"{name}.txt").write_text((" ".join(["1"]*count)+"\n")*2)
    (data/"profile.txt").write_text("100 90 0 130 0\n100 73 0 130 0\n")
    result=pd.read_csv(build_table(data,None,tmp_path/"output"))
    assert result.cycle_id.tolist()==["hydraulic-0001","hydraulic-0002"]
    assert result.valve_condition_percent.tolist()==[90,73]
    assert result.cycle_index.tolist()==[0,1]


def test_oof_schema_and_coverage(trained):
    from sklearn.metrics import accuracy_score, f1_score
    from telemetry_engine.ml.train import CLASSES
    table=pd.read_csv(trained/"table.csv").sort_values("cycle_index")
    oof=pd.read_csv(trained/"oof_predictions.csv")
    assert list(oof)==["cycle_id","cycle_index","fold","p_73","p_80","p_90","p_100","p_degraded"]
    assert oof.cycle_id.is_unique and oof.cycle_id.tolist()==table.cycle_id.tolist()
    for fold,(_,ids) in enumerate(blocked_folds(len(table))):
        assert oof.iloc[ids].fold.eq(fold).all()
    probabilities=oof[[f"p_{c}" for c in CLASSES]].to_numpy()
    assert np.allclose(probabilities.sum(axis=1),1)
    assert np.allclose(oof.p_degraded,1-oof.p_100)
    prediction=np.array(CLASSES)[probabilities.argmax(axis=1)]
    metrics=json.loads((trained/"metrics.json").read_text())["results"]["blocked"]["xgboost"]
    assert accuracy_score(table.valve_condition_percent,prediction)==metrics["accuracy"]
    assert f1_score(table.valve_condition_percent,prediction,average="macro")==metrics["macro_f1"]
