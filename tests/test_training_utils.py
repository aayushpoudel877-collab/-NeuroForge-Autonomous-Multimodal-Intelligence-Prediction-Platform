from neuroforge.deep_learning.training_loop import EarlyStopping

def test_early_stopping():
    e=EarlyStopping(patience=2)
    assert e.step(.5)
    assert not e.step(.4)
    assert e.step(.6)
    assert not e.stopped
    assert not e.step(.5)
    assert e.step(.4)
    assert e.stopped
