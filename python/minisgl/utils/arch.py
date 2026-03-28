from __future__ import annotations

import functools
from typing import Tuple


@functools.cache
def _get_torch_cuda_version() -> Tuple[int, int] | None:
    import torch
    import torch.version

    if not torch.cuda.is_available() or not torch.version.cuda:
        return None
    return torch.cuda.get_device_capability()


def is_arch_supported(major: int, minor: int = 0) -> bool:
    arch = _get_torch_cuda_version()
    if arch is None:
        return False
    return arch >= (major, minor)


def is_sm90_supported() -> bool:
    # Hopper is compute capability 9.x. Do not treat 10.x / 12.x as "SM90+"
    # or FlashAttention will select Hopper kernels and fail on newer GPUs.
    arch = _get_torch_cuda_version()
    if arch is None:
        return False
    major, _minor = arch
    return major == 9


def is_sm100_supported() -> bool:
    # TRTLLM / TllmGen FMHA targets NVIDIA "SM100" (compute capability 10.x).
    # Consumer Blackwell (e.g. RTX 50xx) reports 12.x and must not use trtllm here.
    arch = _get_torch_cuda_version()
    if arch is None:
        return False
    major, _minor = arch
    return major == 10
