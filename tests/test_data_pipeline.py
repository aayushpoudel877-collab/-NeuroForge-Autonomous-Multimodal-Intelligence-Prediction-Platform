import numpy as np
from neuroforge.data.manifest import DatasetManifest,SampleRecord
from neuroforge.data.quality import profile_manifest,validate_manifest
from neuroforge.data.seeding import seed_everything

def make_manifest():
    return DatasetManifest([SampleRecord(str(i),i%2,text="sample") for i in range(10)])

def test_deterministic_split():
    a,_,_=make_manifest().split(seed=7)
    b,_,_=make_manifest().split(seed=7)
    assert [x.sample_id for x in a.records]==[x.sample_id for x in b.records]

def test_profile_and_validation():
    m=make_manifest()
    profile=profile_manifest(m)
    assert profile["samples"]==10
    assert validate_manifest(m)==[]

def test_seed_helper():
    seed_everything(123)


def test_multimodal_batch_collation():
    from neuroforge.deep_learning.dataset import MultimodalDataset
    from neuroforge.data.batch import collate_records
    records=[SampleRecord(str(i),i%2,text="sample") for i in range(2)]
    loader=lambda path: np.ones((2,2),dtype=np.float32)
    dataset=MultimodalDataset(records,array_loader=loader)
    items=[]
    for i in range(2):
        item=dataset[i]
        item["image"]=np.ones((3,8,8),dtype=np.float32)
        item["audio"]=np.ones((1,32),dtype=np.float32)
        item["series"]=np.ones((4,1),dtype=np.float32)
        items.append(item)
    batch=collate_records(items)
    assert batch["text"].shape==(2,128)
    assert batch["image"].shape==(2,3,8,8)
    assert batch["labels"].shape==(2,)
