from __future__ import annotations

def autocast_context(enabled=True,device_type="cuda"):
    import torch
    if not enabled or device_type=="cpu":
        return torch.autocast(device_type=device_type,enabled=False)
    return torch.autocast(device_type=device_type,enabled=True)

def create_grad_scaler(enabled=True):
    import torch
    try:
        return torch.amp.GradScaler("cuda",enabled=enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)
