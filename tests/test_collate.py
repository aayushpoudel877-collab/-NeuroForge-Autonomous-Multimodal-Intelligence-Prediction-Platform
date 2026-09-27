import numpy as np
from neuroforge.deep_learning.collate import collate_samples

class S:
    sample_id="a"; label=1; text=np.zeros(128); image=np.zeros((3,64,64)); audio=np.zeros((1,16000)); series=np.zeros((64,1))

def test_collate():
    batch=collate_samples([S(),S()])
    assert batch["text"].shape==(2,128)
    assert batch["image"].shape==(2,3,64,64)
