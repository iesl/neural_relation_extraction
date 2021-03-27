import warnings
import torch
from typing import *

__all__ = [
    "cuda_if_available",
]

def cuda_if_available(use_cuda: Optional[bool] = None) -> torch.device:
    cuda_available = torch.cuda.is_available()
    _use_cuda = (use_cuda is None or use_cuda) and cuda_available
    if use_cuda is True and not cuda_available:
        warnings.warn("Requested CUDA but it is not available, running on CPU")
    if use_cuda is False and cuda_available:
        warnings.warn(
            "Running on CPU, even though CUDA is available. "
            "(This is likely not desired, check your arguments.)"
        )
    return torch.device("cuda" if _use_cuda else "cpu")