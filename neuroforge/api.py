from fastapi import FastAPI,HTTPException
from .pipeline import NeuroForgePipeline
from .schemas import PredictionRequest,PredictionResponse
app=FastAPI(title="NeuroForge API",version="0.1.0")
pipeline=NeuroForgePipeline()
@app.get("/health")
def health():return {"status":"ok","service":"neuroforge"}
@app.post("/v1/predict",response_model=PredictionResponse)
def predict(request:PredictionRequest):
    try:return pipeline.predict(request)
    except ValueError as e:raise HTTPException(422,str(e)) from e
    except Exception as e:raise HTTPException(500,f"inference failure: {e}") from e
