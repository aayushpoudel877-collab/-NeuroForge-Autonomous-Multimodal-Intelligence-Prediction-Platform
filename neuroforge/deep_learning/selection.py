from __future__ import annotations

def select_best(runs,metric="f1",maximize=True):
    completed=[r for r in runs if r.get("status")=="completed" and r.get("metrics") and metric in r["metrics"]]
    if not completed:return None
    return sorted(completed,key=lambda r:r["metrics"][metric],reverse=maximize)[0]

def promotion_candidate(runs,metric="f1",minimum=0.0):
    best=select_best(runs,metric)
    if best is None:return None
    return best if best["metrics"][metric]>=minimum else None
