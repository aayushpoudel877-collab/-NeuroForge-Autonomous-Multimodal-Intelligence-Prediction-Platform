from __future__ import annotations
import os
from fastapi import Header,HTTPException
from .security import verify_api_key

def require_api_key(x_api_key: str|None=Header(default=None,alias="X-API-Key")):
    required=os.getenv("NEUROFORGE_REQUIRE_API_KEY","false").lower()=="true"
    expected=os.getenv("NEUROFORGE_API_KEY_HASH")
    if not required:
        return
    if not expected or not x_api_key or not verify_api_key(x_api_key,expected):
        raise HTTPException(status_code=401,detail="authentication required")
