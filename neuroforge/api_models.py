from pydantic import BaseModel,Field

class BatchPredictionRequest(BaseModel):
    requests:list[dict]=Field(min_length=1,max_length=128)

class ModelInfo(BaseModel):
    name:str
    version:str
    stage:str

class ReadinessResponse(BaseModel):
    ready:bool
    checks:dict[str,bool]
