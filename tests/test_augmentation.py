import numpy as np
from neuroforge.data.augment import add_gaussian_noise,random_mask

def test_noise_preserves_shape():
    x=np.ones((4,8))
    assert add_gaussian_noise(x).shape==x.shape

def test_mask_preserves_shape():
    x=np.ones((4,8))
    y=random_mask(x,.2)
    assert y.shape==x.shape
    assert np.isfinite(y).all()
