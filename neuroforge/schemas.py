from typing import Any
from pydantic import BaseModel,Field

class PredictionRequest(BaseModel):
    text:str|None=None
    image_base64:str|None=None
    audio_base64:str|None=None
    series:list[float]|None=Field(default=None,min_length=3)
    metadata:dict[str,Any]=Field(default_factory=dict)

class ModalityResult(BaseModel):
    modality:str
    score:float
    confidence:float
    features:list[float]=Field(default_factory=list)

class PredictionResponse(BaseModel):
    prediction:str
    probability:float
    confidence:float
    modalities:list[ModalityResult]
    explanation:dict[str,Any]
    pipeline_ms:float
