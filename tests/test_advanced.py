import pytest
pytest.importorskip("torch")
from neuroforge.deep_learning.pretrained import HuggingFaceTextEncoder,TorchvisionVisionEncoder,Wav2Vec2AudioEncoder

def test_advanced_backbone_adapters_are_importable():
    assert HuggingFaceTextEncoder.__name__
    assert TorchvisionVisionEncoder.__name__
    assert Wav2Vec2AudioEncoder.__name__
