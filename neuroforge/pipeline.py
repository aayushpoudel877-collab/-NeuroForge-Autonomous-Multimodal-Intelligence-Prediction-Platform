import time,base64
from .fusion import ConfidenceWeightedFusion
from .models.text import TextEncoder
from .models.image import ImageEncoder
from .models.audio import AudioEncoder
from .models.timeseries import TimeSeriesEncoder
from .explainability import ExplanationEngine
from .schemas import PredictionRequest,PredictionResponse,ModalityResult
from .data.validation import validate_request
class NeuroForgePipeline:
    def __init__(self):
        self.text,self.image,self.audio,self.series=TextEncoder(),ImageEncoder(),AudioEncoder(),TimeSeriesEncoder()
        self.fusion,self.explainer=ConfidenceWeightedFusion(),ExplanationEngine()
    def predict(self,r):
        t=time.perf_counter();ok,errors=validate_request(r.text,r.image_base64,r.audio_base64,r.series)
        if not ok:raise ValueError("; ".join(errors))
        results=[];out=[];anomaly=0
        def add(name,s,c,f):results.append((name,s,c));out.append(ModalityResult(modality=name,score=s,confidence=c,features=f))
        if r.text:add("text",*self.text.predict(r.text))
        if r.image_base64:add("image",*self.image.predict(base64.b64decode(r.image_base64)))
        if r.audio_base64:add("audio",*self.audio.predict(base64.b64decode(r.audio_base64)))
        if r.series:
            s,c,f=self.series.predict(r.series);anomaly=self.series.anomaly_score(r.series);add("timeseries",s,c,f)
        fused=self.fusion.fuse(results)
        return PredictionResponse(prediction="class_1" if fused.probability>=.5 else "class_0",probability=fused.probability,confidence=fused.confidence,modalities=out,explanation=self.explainer.build(fused.probability,fused.confidence,fused.weights,anomaly),pipeline_ms=(time.perf_counter()-t)*1000)
