class ExplanationEngine:
    def build(self,probability,confidence,weights,anomaly=0):
        ordered=sorted(weights.items(),key=lambda x:x[1],reverse=True)
        evidence=[f"{n} contributed {w:.1%}" for n,w in ordered]
        if anomaly>.65:evidence.append(f"time-series anomaly score is elevated at {anomaly:.2f}")
        return {"summary":f"Prediction leans {'positive' if probability>=.5 else 'negative'} with {confidence:.1%} confidence.","evidence":evidence,"uncertainty":1-confidence,"fusion_weights":weights,"anomaly_score":anomaly}
