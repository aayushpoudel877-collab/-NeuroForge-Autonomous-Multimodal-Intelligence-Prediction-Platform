import numpy as np
from neuroforge.data.preprocessors import tokenize_text,prepare_image,prepare_audio,prepare_series

def test_preprocessors_contract():
    assert tokenize_text("hello").shape==(128,)
    assert prepare_image(np.ones((8,8,3)),16).shape==(3,16,16)
    assert prepare_audio(np.ones(20),32).shape==(1,32)
    assert prepare_series(np.arange(20),16).shape==(16,1)
