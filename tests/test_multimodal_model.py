import pytest

torch=pytest.importorskip("torch")
from neuroforge.deep_learning import NeuroForgeMultimodalModel

def test_unified_multimodal_forward():
    model=NeuroForgeMultimodalModel(embedding_dim=32)
    text=torch.randint(0,4096,(2,16))
    images=torch.randn(2,3,64,64)
    audio=torch.randn(2,1,256)
    series=torch.randn(2,20,1)
    output=model(text,images,audio,series)
    assert output.shape==(2,)
    assert torch.isfinite(output).all()

def test_cross_modal_attention_shape():
    from neuroforge.deep_learning import CrossModalAttention
    block=CrossModalAttention(embedding_dim=32)
    tokens=[torch.randn(2,32) for _ in range(4)]
    output=block(tokens)
    assert output.shape==(2,4,32)
