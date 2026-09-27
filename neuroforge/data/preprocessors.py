from __future__ import annotations
import numpy as np

def tokenize_text(text: str, max_length: int = 128) -> np.ndarray:
    values=np.frombuffer((text or "").encode("utf-8"),dtype=np.uint8).astype(np.int64)
    values=np.minimum(values,4095)[:max_length]
    return np.pad(values,(0,max_length-len(values)))

def prepare_image(array, size: int = 64) -> np.ndarray:
    x=np.asarray(array,dtype=np.float32)
    if x.ndim==2: x=x[...,None]
    if x.ndim!=3: raise ValueError("image must be HxWxC or HxW")
    if x.shape[-1] not in (1,3): raise ValueError("image channel count must be 1 or 3")
    if x.shape[-1]==1: x=np.repeat(x,3,axis=-1)
    x=x/255.0 if x.max()>1.0 else x
    h,w=x.shape[:2]
    yy=np.linspace(0,h-1,size).astype(int); xx=np.linspace(0,w-1,size).astype(int)
    return np.transpose(x[yy][:,xx],(2,0,1)).astype(np.float32)

def prepare_audio(array, max_samples: int = 16000) -> np.ndarray:
    x=np.asarray(array,dtype=np.float32).reshape(-1)[:max_samples]
    if len(x)<max_samples: x=np.pad(x,(0,max_samples-len(x)))
    scale=max(float(np.max(np.abs(x))),1e-6)
    return (x/scale)[None,:].astype(np.float32)

def prepare_series(array, length: int = 64, features: int = 1) -> np.ndarray:
    x=np.asarray(array,dtype=np.float32)
    if x.ndim==1: x=x[:,None]
    if x.ndim!=2 or x.shape[1]!=features: raise ValueError("series must be [time, features]")
    idx=np.linspace(0,len(x)-1,length).astype(int)
    y=x[idx]
    mean=y.mean(axis=0,keepdims=True); std=y.std(axis=0,keepdims=True)
    return ((y-mean)/np.maximum(std,1e-6)).astype(np.float32)
