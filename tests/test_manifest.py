from neuroforge.data.manifest import DatasetManifest,SampleRecord

def test_small_manifest_split_is_non_empty():
    manifest=DatasetManifest([SampleRecord(str(i),i%2) for i in range(4)])
    train,val,test=manifest.split()
    assert len(train.records)>=1 and len(val.records)>=1 and len(test.records)>=1
