from fastapi import FastAPI,HTTPException
from .pipeline import NeuroForgePipeline
from .schemas import PredictionRequest,PredictionResponse
from .api_models import BatchPredictionRequest,ModelInfo,ReadinessResponse
from .monitoring import PredictionMonitor
from .health import ServiceHealth
from .api_metrics import ApiMetrics
from .registry import JsonModelRegistry

app=FastAPI(title="NeuroForge API",version="0.2.0")
pipeline=NeuroForgePipeline()
monitor=PredictionMonitor()
health_state=ServiceHealth()
metrics=ApiMetrics()
registry=JsonModelRegistry()

@app.get("/health")
def health():
    return health_state.snapshot()

@app.get("/ready",response_model=ReadinessResponse)
def ready():
    checks={"pipeline":pipeline is not None,"monitor":monitor is not None,"registry":registry is not None}
    return ReadinessResponse(ready=all(checks.values()),checks=checks)

@app.get("/metrics")
def api_metrics():
    return {"api":metrics.snapshot(),"inference":monitor.snapshot(),"health":health_state.snapshot()}

@app.get("/v1/models",response_model=list[ModelInfo])
def models():
    return [ModelInfo(name=x["name"],version=x["version"],stage=x["stage"]) for x in registry.list()]

@app.post("/v1/predict",response_model=PredictionResponse)
def predict(request:PredictionRequest):
    try:
        response=pipeline.predict(request)
        monitor.record(response.probability,response.pipeline_ms)
        metrics.prediction()
        health_state.request()
        return response
    except ValueError as e:
        metrics.prediction(True);health_state.request(True)
        raise HTTPException(422,str(e)) from e
    except Exception as e:
        metrics.prediction(True);health_state.request(True)
        raise HTTPException(500,f"inference failure: {e}") from e

@app.post("/v1/batch-predict",response_model=list[PredictionResponse])
def batch_predict(request:BatchPredictionRequest):
    metrics.batch()
    results=[]
    for item in request.requests:
        results.append(predict(PredictionRequest.model_validate(item)))
    return results
