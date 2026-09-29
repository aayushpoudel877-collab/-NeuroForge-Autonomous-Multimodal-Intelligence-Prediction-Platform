import pytest
torch=pytest.importorskip("torch")
from neuroforge.deep_learning.pretrained import HuggingFaceTextEncoder,TorchvisionVisionEncoder,Wav2Vec2AudioEncoder

def test_advanced_backbones_fail_cleanly_when_dependency_missing():
    for factory in (HuggingFaceTextEncoder,TorchvisionVisionEncoder,Wav2Vec2AudioEncoder):
        try: obj=factory()
        except ImportError: continue
        assert obj is not None
