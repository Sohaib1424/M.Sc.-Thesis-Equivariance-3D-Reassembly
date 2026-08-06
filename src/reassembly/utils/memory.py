"""
Memory budgeting: estimating what a batch will cost before running it, and
recovering cleanly when the estimate is wrong.

WHY THIS MODULE EXISTS
----------------------
The original failure was "training crashes with CUDA OOM partway through an
epoch on 2x16 GB T4s". The intermittency is the diagnostic clue: Breaking Bad
fragment resolution varies by more than an order of magnitude, so peak memory
is set by whichever scene the sampler happens to draw. Capping ``batch_size``
low enough to survive the worst draw wastes the GPU on every other batch.

Three complementary mitigations, in order of how much they buy:

  1. A vertex budget at the data layer (``data/decimate.py``) -- makes peak
     memory a number you choose rather than one the dataset chooses.
  2. Gradient checkpointing plus mixed precision in the model/loop.
  3. This module: a cheap pre-flight estimate that skips a batch which is
     obviously too large, and an OOM handler that recovers instead of dying.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

BYTES_PER_FP32 = 4
BYTES_PER_FP16 = 2


@dataclass
class MemoryEstimate:
    activation_bytes: int
    dominant_term: str

    @property
    def gib(self) -> float:
        return self.activation_bytes / 2**30


def estimate_activation_bytes(
    num_nodes: int,
    num_directed_edges: int,
    hidden_channels: int,
    num_layers: int,
    num_vn_slots: int = 8,
    amp: bool = False,
    gradient_checkpointing: bool = False,
) -> MemoryEstimate:
    """Rough peak activation footprint of one forward+backward.

    Counts the two tensors that actually dominate, and deliberately ignores
    everything small:

      mesh attention   ~5 tensors of (2E, hidden, 3)   per layer
      virtual nodes    ~4 tensors of (N, hidden, 3) + (N, heads, K)  per layer

    Accurate to maybe a factor of 1.5, which is enough for its job: deciding
    whether a batch is *obviously* too big. It is not a substitute for
    ``torch.cuda.max_memory_allocated()``, which
    ``scripts/profile_memory.py`` reports for real.
    """
    width = BYTES_PER_FP16 if amp else BYTES_PER_FP32

    mesh = 5 * num_directed_edges * hidden_channels * 3 * width
    vnode = 4 * num_nodes * hidden_channels * 3 * width + num_nodes * num_vn_slots * 4 * width

    per_layer = mesh + vnode
    if gradient_checkpointing:
        # Only the stage boundaries are stored; one stage is recomputed at a
        # time during backward, so the transient peak is one layer's worth.
        total = num_layers * num_nodes * hidden_channels * 3 * width + per_layer
        dominant = "checkpointed (one stage recomputed at a time)"
    else:
        total = num_layers * per_layer
        dominant = "mesh messages" if mesh > vnode else "virtual-node attention"

    return MemoryEstimate(int(total), dominant)


def cuda_memory_summary(device: Optional[torch.device] = None) -> str:
    """Allocated / reserved / peak, guarded so it can never mask a real error."""
    if device is None or device.type != "cuda":
        return ""
    try:
        return (
            f"allocated={torch.cuda.memory_allocated(device) / 2**30:.2f} GiB, "
            f"reserved={torch.cuda.memory_reserved(device) / 2**30:.2f} GiB, "
            f"peak={torch.cuda.max_memory_allocated(device) / 2**30:.2f} GiB"
        )
    except RuntimeError:
        # If the original error already poisoned the CUDA context, even this
        # query throws -- and if that happened unguarded it would become the
        # exception that propagates, burying the real one under a confusing
        # "during handling of the above exception" chain.
        return "(memory stats unavailable -- CUDA context likely already broken)"


def is_oom_error(exc: BaseException) -> bool:
    """Is this exception an out-of-memory condition?

    Matches on message text because PyTorch raises a plain ``RuntimeError`` for
    CUDA OOM on many versions rather than a dedicated class. Both the CUDA and
    CPU wordings are covered.
    """
    if isinstance(exc, torch.cuda.OutOfMemoryError):  # type: ignore[attr-defined]
        return True
    msg = str(exc).lower()
    return ("out of memory" in msg) or ("cuda oom" in msg) or ("can't allocate memory" in msg)


def release_cuda_memory() -> None:
    """Drop cached blocks after an OOM so the next batch starts from a clean
    allocator state. Without this, fragmentation from the failed batch makes
    the *next* batch far more likely to fail too, and a single unlucky draw
    cascades into the whole epoch dying."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


class AmpContext:
    """Autocast + GradScaler, or transparent no-ops when AMP is off.

    T4s are Turing: they have fast fp16 tensor cores but NO bf16 support, so
    fp16 with loss scaling is the correct choice here rather than bf16. On
    Ampere or newer, ``dtype='bf16'`` avoids needing a scaler at all.
    """

    def __init__(self, enabled: bool, device_type: str = "cuda", dtype: str = "fp16"):
        self.enabled = enabled and device_type == "cuda"
        self.device_type = device_type
        self.dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
        needs_scaler = self.enabled and self.dtype is torch.float16
        try:
            self.scaler = torch.amp.GradScaler(device_type, enabled=needs_scaler)
        except (AttributeError, TypeError):  # older torch
            self.scaler = torch.cuda.amp.GradScaler(enabled=needs_scaler)

    def autocast(self):
        return torch.autocast(
            device_type=self.device_type, dtype=self.dtype, enabled=self.enabled
        )

    def backward(self, loss: torch.Tensor) -> None:
        self.scaler.scale(loss).backward()

    def step(self, optimizer, grad_clip: Optional[float] = None, parameters=None) -> None:
        if grad_clip is not None and grad_clip > 0:
            # Unscale first, or the clip threshold would be applied to
            # loss-scaled gradients and effectively do nothing.
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, grad_clip)
        self.scaler.step(optimizer)
        self.scaler.update()
