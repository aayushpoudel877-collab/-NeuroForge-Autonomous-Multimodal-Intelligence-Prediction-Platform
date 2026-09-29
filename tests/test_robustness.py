import numpy as np
import pytest
torch=pytest.importorskip("torch")
from neuroforge.deep_learning.unified_model import NeuroForgeMultimodalModel
from neuroforge.deep_learning.robustness import random_mask_batches,robustness_report

def _batch():
    return [{"text":np.zeros((2,128),dtype=np.int64),"image":np.zeros((2,3,64,64),dtype=np.float32),
             "audio":np.zeros((2,1,64),dtype=np.float32),"series":np.zeros((2,64,1),dtype=np.float32),
             "labels":np.array([0,1],dtype=np.float32)}]

def test_random_masks_keep_one_modality():
    masked=random_mask_batches(_batch(),.9,7)[0]["modality_mask"]
    assert masked.shape==(2,4) and masked.any(axis=1).all()

def test_robustness_report_has_multiple_levels():
    model=NeuroForgeMultimodalModel()
    report=robustness_report(model,_batch(),(.0,.5),seed=7)
    assert set(report)=={"0.0","0.5"}
