from __future__ import annotations
import numpy as np

def add_gaussian_noise(values,scale=.01,rng=None):
    array=np.asarray(values,dtype=float)
    generator=rng or np.random.default_rng()
    return array+generator.normal(0,scale,size=array.shape)

def random_mask(values,rate=.1,rng=None):
    if not 0<=rate<1: raise ValueError("rate must be in [0,1)")
    array=np.asarray(values).copy()
    generator=rng or np.random.default_rng()
    mask=generator.random(array.shape)<rate
    array[mask]=0
    return array
