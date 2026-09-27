from __future__ import annotations
from dataclasses import dataclass
from itertools import product
from .runs import RunConfig

@dataclass(frozen=True)
class SearchSpace:
    learning_rates:tuple[float,...]=(1e-3,3e-4)
    batch_sizes:tuple[int,...]=(16,32)
    epochs:tuple[int,...]=(5,10)

def grid_configs(space:SearchSpace,base_seed=42):
    for lr,bs,epochs in product(space.learning_rates,space.batch_sizes,space.epochs):
        yield RunConfig(learning_rate=lr,batch_size=bs,epochs=epochs,seed=base_seed)
