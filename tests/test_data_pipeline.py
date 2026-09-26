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
