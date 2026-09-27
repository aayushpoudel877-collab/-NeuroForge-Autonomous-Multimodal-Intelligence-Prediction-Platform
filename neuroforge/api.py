from __future__ import annotations
from time import perf_counter
from fastapi import FastAPI,HTTPException,Header,Response
from .pipeline import NeuroForgePipeline
from .schemas import PredictionRequest,PredictionResponse
from .api_models import BatchPredictionRequest,ModelInfo,ReadinessResponse
from .monitoring import PredictionMonitor
from .health import ServiceHealth
from .api_metrics import ApiMetrics
from .registry import JsonModelRegistry
from .observability import InferenceTrace,TraceStore,new_request_id,utc_now
from .alerts import AlertStore,threshold_alert

app=FastAPI(title="NeuroForge API",version="0.3.0")
pipeline=NeuroForgePipeline()
monitor=PredictionMonitor()
health_state=ServiceHealth()
metrics=ApiMetrics()
registry=JsonModelRegistry()
traces=TraceStore()
alerts=AlertStore()

@app.get("/health")
def health():
    return health_state.snapshot()

@app.get("/ready",response_model=ReadinessResponse)
def ready():
    checks={"pipeline":pipeline is not None,"monitor":monitor is not None,"registry":registry is not None,"trace_store":traces is not None}
    return ReadinessResponse(ready=all(checks.values()),checks=checks)

@app.get("/metrics")
def api_metrics():
    return {"api":metrics.snapshot(),"inference":monitor.snapshot(),"health":health_state.snapshot()}

@app.get("/metrics/prometheus")
def prometheus_metrics():
    api=metrics.snapshot();health=health_state.snapshot();inf=monitor.snapshot()
    lines=[
        "# HELP neuroforge_predictions_total Total prediction requests.",
        "# TYPE neuroforge_predictions_total counter",
        f"neuroforge_predictions_total {api['predictions_total']}",
        "# HELP neuroforge_prediction_failures_total Failed prediction requests.",
        "# TYPE neuroforge_prediction_failures_total counter",
        f"neuroforge_prediction_failures_total {api['prediction_failures_total']}",
        "# HELP neuroforge_batches_total Total batch requests.",
        "# TYPE neuroforge_batches_total counter",
        f"neuroforge_batches_total {api['batches_total']}",
        "# HELP neuroforge_prediction_failure_rate Prediction failure ratio.",
        "# TYPE neuroforge_prediction_failure_rate gauge",
        f"neuroforge_prediction_failure_rate {api['failure_rate']}",
        "# HELP neuroforge_mean_latency_ms Mean inference latency over the active window.",
        "# TYPE neuroforge_mean_latency_ms gauge",
        f"neuroforge_mean_latency_ms {inf['mean_latency_ms']}",
        "# HELP neuroforge_health_error_rate Service error ratio.",
        "# TYPE neuroforge_health_error_rate gauge",
        f"neuroforge_health_error_rate {health['error_rate']}",
    ]
    return Response("\n".join(lines)+"\n",media_type="text/plain; version=0.0.4")

@app.get("/observability/traces")
def recent_traces(limit:int=100):
    return traces.list(limit)

@app.get("/observability/alerts")
def recent_alerts(limit:int=100):
    return alerts.recent(limit)

@app.get("/v1/models",response_model=list[ModelInfo])
def models():
    return [ModelInfo(name=x["name"],version=x["version"],stage=x["stage"]) for x in registry.list()]

@app.post("/v1/predict",response_model=PredictionResponse)
def predict(request:PredictionRequest,response:Response,request_id:str|None=Header(default=None,alias="X-Request-ID")):
    rid=request_id or new_request_id()
    started=utc_now()
    started_clock=perf_counter()
    try:
        result=pipeline.predict(request)
        monitor.record(result.probability,result.pipeline_ms)
        metrics.prediction()
        health_state.request()
        response.headers["X-Request-ID"]=rid
        latency=(perf_counter()-started_clock)*1000
        trace=InferenceTrace(
            request_id=rid,started_at=started,finished_at=utc_now(),status="ok",
            latency_ms=latency,modalities=[m.modality for m in result.modalities],
            prediction=result.prediction,probability=result.probability,
            model_name=request.metadata.get("model_name"),
            model_version=request.metadata.get("model_version"),
            dataset_snapshot=request.metadata.get("dataset_snapshot"),
            training_run=request.metadata.get("training_run"),
            artifact_sha256=request.metadata.get("artifact_sha256"),
        )
        try: traces.append(trace)
        except Exception: pass
        alert=threshold_alert("latency",latency,2000.0,"inference latency exceeded 2000 ms")
        if alert:
            try: alerts.emit(alert)
            except Exception: pass
        return result
    except ValueError as e:
        metrics.prediction(True);health_state.request(True)
        response.headers["X-Request-ID"]=rid
        try:
            traces.append(InferenceTrace(rid,started,utc_now(),"error",(perf_counter()-started_clock)*1000,error_type=type(e).__name__))
        except Exception: pass
        raise HTTPException(422,str(e)) from e
    except Exception as e:
        metrics.prediction(True);health_state.request(True)
        response.headers["X-Request-ID"]=rid
        try:
            traces.append(InferenceTrace(rid,started,utc_now(),"error",(perf_counter()-started_clock)*1000,error_type=type(e).__name__))
        except Exception: pass
        raise HTTPException(500,"inference failure") from e

@app.post("/v1/batch-predict",response_model=list[PredictionResponse])
def batch_predict(request:BatchPredictionRequest):
    metrics.batch()
    return [predict(item,Response()) for item in request.requests]
