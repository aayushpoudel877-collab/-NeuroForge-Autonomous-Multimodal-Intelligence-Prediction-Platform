from neuroforge.observability import InferenceTrace,TraceStore,new_request_id
from neuroforge.alerts import AlertStore,threshold_alert

def test_trace_store(tmp_path):
    store=TraceStore(tmp_path/"traces.jsonl")
    trace=InferenceTrace(new_request_id(),"a","b","ok",1.2,["text"],"class_1",.8)
    store.append(trace)
    assert store.list()[0]["status"]=="ok"

def test_alert_store(tmp_path):
    store=AlertStore(tmp_path/"alerts.jsonl")
    alert=threshold_alert("drift",.3,.2,"drift detected")
    store.emit(alert)
    assert store.recent()[0]["alert_type"]=="drift"
