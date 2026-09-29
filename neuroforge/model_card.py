from __future__ import annotations
from pathlib import Path
import json

def build_model_card(model_name,task,metrics,modalities,limitations=None):
    return {"model_name":model_name,"task":task,"modalities":list(modalities),
            "metrics":metrics,"limitations":list(limitations or []),
            "disclaimer":"Metrics describe the supplied evaluation data only; they are not evidence of real-world performance."}

def save_model_card(card,path="reports/model-card.json"):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(card,indent=2)); return str(p)
