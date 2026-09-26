from __future__ import annotations
from pathlib import Path

class CheckpointManager:
    def __init__(self,directory="models/checkpoints"):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
    def save(self,model,optimizer,epoch,metric,name="latest.pt"):
        import torch
        path=self.directory/name
        torch.save({"epoch":epoch,"metric":metric,"model":model.state_dict(),"optimizer":optimizer.state_dict()},path)
        return str(path)
    def load(self,path,model,optimizer=None,map_location="cpu"):
        import torch
        state=torch.load(path,map_location=map_location,weights_only=False)
        model.load_state_dict(state["model"])
        if optimizer is not None: optimizer.load_state_dict(state["optimizer"])
        return state
