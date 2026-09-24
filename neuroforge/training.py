import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score
def make_dataset(n=512,seed=42):
    rng=np.random.default_rng(seed);X=rng.normal(size=(n,32));y=(X[:,:4].mean(1)>0).astype(int);return X,y
def train_demo(seed=42):
    X,y=make_dataset(seed=seed);a,b,c,d=train_test_split(X,y,test_size=.2,random_state=seed,stratify=y)
    m=LogisticRegression(max_iter=500,random_state=seed).fit(a,c);p=m.predict_proba(b)[:,1]
    return m,{"accuracy":accuracy_score(d,p>.5),"f1":f1_score(d,p>.5),"roc_auc":roc_auc_score(d,p)}
if __name__=="__main__":print(train_demo()[1])
