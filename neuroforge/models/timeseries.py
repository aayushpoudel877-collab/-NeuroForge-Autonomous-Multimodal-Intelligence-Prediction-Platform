import numpy as np
class TimeSeriesEncoder:
    def predict(self,series):
        x=np.asarray(series,dtype=float);slope=float(np.polyfit(np.arange(len(x)),x,1)[0]);std=float(x.std());z=float((x[-1]-x.mean())/(std+1e-8))
        score=float(1/(1+np.exp(-.8*z-.5*slope)));conf=float(np.clip(.55+min(abs(z)*.12,.35),0,.95))
        return score,conf,[float(x[-1]),float(x.mean()),slope,float(np.std(np.diff(x))),z,float(x.min()),float(x.max()),float(np.median(x))]
    def anomaly_score(self,series):
        x=np.asarray(series,dtype=float);z=abs((x[-1]-x[:-1].mean())/(x[:-1].std()+1e-8))
        return float(np.clip(z/6,0,1))
