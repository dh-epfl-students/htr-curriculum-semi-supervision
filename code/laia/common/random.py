from __future__ import absolute_import

import random
from typing import Optional

import numpy as np
import torch


def manual_seed(seed):
    # type: (int) -> None
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    #torch.cuda.manual_seed(seed)
    #torch.cuda.manual_seed_all(seed)
    #torch.backends.cudnn.enabled = False 
    #torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.deterministic = True


def get_rng_state(device=None):
    # type: (Optional[torch.Device]) -> dict
    torch_cuda_rng_state = (
        torch.cuda.get_rng_state(device=device.index) if device.type == "cuda" else None
    )
    return {
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch_cuda_rng_state,
        "python": random.getstate(),
    }


def set_rng_state(state, device=None):
    # type: (dict, Optional[torch.Device]) -> None
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"] is not None:
        torch.cuda.set_rng_state(
            state["torch_cuda"].cpu(), device=device.index if device else -1
        )
    random.setstate(state["python"])
