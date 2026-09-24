import re,numpy as np
POS={"good","growth","safe","improve","rising","success","stable","positive"}
NEG={"bad","risk","failure","falling","unsafe","decline","negative","critical"}
class TextEncoder:
    def encode(self,text):
        t=re.findall(r"[a-zA-Z]+",text.lower())
        if not t:return np.zeros(8,dtype=np.float32)
        p=sum(x in POS for x in t);n=sum(x in NEG for x in t)
        return np.array([len(t),len(set(t)),p,n,p-n,np.mean([len(x) for x in t]),hash(t[0])%997/997,hash(t[-1])%991/991],dtype=np.float32)
    def predict(self,text):
        f=self.encode(text);s=float(1/(1+np.exp(-(f[4]+.02*f[0]))));c=float(min(.99,.5+abs(s-.5)))
        return s,c,f.tolist()
