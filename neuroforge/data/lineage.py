from __future__ import annotations
from dataclasses import dataclass,asdict
from datetime import datetime,timezone
import hashlib,json

@dataclass(frozen=True)
class DatasetSnapshot:
    name:str
    version:str
    row_count:int
    schema_hash:str
    created_at:str

def schema_fingerprint(schema:dict)->str:
    payload=json.dumps(schema,sort_keys=True,separators=(",",":")).encode()
    return hashlib.sha256(payload).hexdigest()

def snapshot(name,version,row_count,schema):
    return DatasetSnapshot(name,version,row_count,schema_fingerprint(schema),datetime.now(timezone.utc).isoformat())
