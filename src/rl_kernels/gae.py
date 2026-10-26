"""Triton kernels and PyTorch primitives for Generalized Advantage Estimation (GAE).

References:
    Schulman et al. (2015). High-Dimensional Continuous Control Using Generalized
    Advantage Estimation. https://arxiv.org/abs/1506.02438
"""

from beartype import beartype
from jaxtyping import Float
from jaxtyping import Int64
from jaxtyping import jaxtyped
import torch
from torch import Tensor

from rl_kernels.triton_helper import TRITON_AVAILABLE

if TRITON_AVAILABLE:  # pragma: no cover
    import triton  # ty: ignore[unresolved-import]  # noqa: F401
    import triton.language as tl  # ty: ignore[unresolved-import]  # noqa: F401

    @triton.jit
    def _fused_gae_kernel(
        rewards_ptr,
        values_ptr,
        next_values_ptr,
        dones_ptr,
        seq_lens_ptr,
        adv_ptr,
        target_values_ptr,
        stride_b,
        gamma: tl.constexpr,
        lam: tl.constexpr,
    ):
        """Fused Triton GAE Kernel.

        Program instance per batch row, backward pass in registers.
        """
        pid = tl.program_id(axis=0)
        offset = pid * stride_b

        next_value = tl.load(next_values_ptr + pid)
        seq_len = tl.load(seq_lens_ptr + pid).to(tl.int32)

        last_gae = 0.0
        gamma_lam = gamma * lam

        for t in range(seq_len - 1, -1, -1):
            offset_t = offset + t

            reward = tl.load(rewards_ptr + offset_t)
            value = tl.load(values_ptr + offset_t)
            done = tl.load(dones_ptr + offset_t)

            non_terminal_mask = 1 - done
            delta = reward + gamma * next_value * non_terminal_mask - value

            last_gae = delta + gamma_lam * non_terminal_mask * last_gae
            target_value = last_gae + value

            tl.store(adv_ptr + offset_t, last_gae)
            tl.store(target_values_ptr + offset_t, target_value)

            next_value = value

    @jaxtyped(typechecker=beartype)
    def compute_gae_triton[B, T](
        rewards: Float[Tensor, "B T"],
        values: Float[Tensor, "B T"],
        next_values: Float[Tensor, "B"],
        dones: Float[Tensor, "B T"],
        seq_lens: Int64[Tensor, "B"],
        gamma: float = 0.99,
        lam: float = 0.95,
    ) -> tuple[
        Float[Tensor, "B T"],
        Float[Tensor, "B T"],
    ]:
        """Computes Generalized Advantage Estimation (GAE) using Triton fused kernel.

        Returns a tuple of `(advantages, target_values)` where:
            - advantages: Advantage estimates computed backward in registers.
            - target_values: TD(lambda) return targets (`advantages + values`) for
              critic training.
        """
        batch_size, seq_len = rewards.shape
        if values.shape != (batch_size, seq_len):
            raise ValueError("Shape mismatch: values must have shape (B, T).")
        if next_values.shape != (batch_size,):
            raise ValueError("Shape mismatch: next_values must have shape (B,).")
        if dones.shape != (batch_size, seq_len):
            raise ValueError("Shape mismatch: dones must have shape (B, T).")

        if (seq_lens > seq_len).any():
            raise ValueError("Maximum value in seq_lens exceeds sequence length.")

        rewards = rewards.contiguous()
        values = values.contiguous()
        next_values = next_values.contiguous()
        dones = dones.contiguous()
        seq_lens = seq_lens.to(dtype=torch.int32).contiguous()

        advantages = torch.zeros_like(rewards)
        target_values = torch.zeros_like(rewards)

        stride_b = rewards.stride(0)
        grid = (batch_size,)

        _fused_gae_kernel[grid](
            rewards,
            values,
            next_values,
            dones,
            seq_lens,
            advantages,
            target_values,
            stride_b,
            gamma,  # ty: ignore[invalid-argument-type]
            lam,  # ty: ignore[invalid-argument-type]
        )

        return advantages, target_values
else:

    @jaxtyped(typechecker=beartype)
    def compute_gae_triton[B, T](
        rewards: Float[Tensor, "B T"],
        values: Float[Tensor, "B T"],
        next_values: Float[Tensor, "B"],
        dones: Float[Tensor, "B T"],
        seq_lens: Int64[Tensor, "B"],
        gamma: float = 0.99,
        lam: float = 0.95,
    ) -> tuple[
        Float[Tensor, "B T"],
        Float[Tensor, "B T"],
    ]:
        """Stub for Triton GAE when Triton is not available."""
        raise RuntimeError("Triton is not available in the current environment.")


@jaxtyped(typechecker=beartype)
def compute_gae_torch[B, T](
    rewards: Float[Tensor, "B T"],
    values: Float[Tensor, "B T"],
    next_values: Float[Tensor, "B"],
    dones: Float[Tensor, "B T"],
    seq_lens: Int64[Tensor, "B"],
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[
    Float[Tensor, "B T"],
    Float[Tensor, "B T"],
]:
    """Computes Generalized Advantage Estimation (GAE) using PyTorch primitives.

    Follows Schulman et al. (2015), Eq. (11)-(12) with sequence masking.

    Returns a tuple of `(advantages, target_values)` where:
        - advantages: Advantage estimates masked beyond completion lengths.
        - target_values: TD(lambda) return targets (`advantages + values`) for critic
          training.
    """
    max_seq_len = rewards.shape[-1]
    if max_seq_len == 0:
        raise ValueError("Maximum sequence length is zero.")

    if (seq_lens > max_seq_len).any():
        raise ValueError("Maximum value in seq_lens exceeds rewards sequence length.")

    last_gae = torch.zeros_like(rewards[..., 0])
    advantages = torch.zeros_like(rewards)
    dones = dones.to(dtype=rewards.dtype)
    seq_lens = seq_lens.to(device=rewards.device, dtype=torch.int64)

    values_ext = torch.cat([values, next_values.unsqueeze(1)], dim=1)
    values_ext.scatter_(
        dim=1, index=seq_lens.unsqueeze(1), src=next_values.unsqueeze(1)
    )

    for t in reversed(range(max_seq_len)):
        is_valid = t < seq_lens

        delta = (
            rewards[..., t]
            + gamma * (1 - dones[..., t]) * values_ext[..., t + 1]
            - values[..., t]
        )

        last_gae = (delta + gamma * lam * (1 - dones[..., t]) * last_gae) * is_valid
        advantages[..., t] = last_gae

    mask = torch.arange(max_seq_len, device=rewards.device) < seq_lens.unsqueeze(1)
    target_values = (advantages + values) * mask
    return advantages, target_values


@jaxtyped(typechecker=beartype)
def compute_gae[B, T](
    rewards: Float[Tensor, "B T"],
    values: Float[Tensor, "B T"],
    next_values: Float[Tensor, "B"],
    dones: Float[Tensor, "B T"],
    seq_lens: Int64[Tensor, "B"],
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[
    Float[Tensor, "B T"],
    Float[Tensor, "B T"],
]:
    """Computes Generalized Advantage Estimation (GAE) across variable-length rollouts.

    Dispatches to fused Triton GPU kernel when Triton and an accelerator device
    are available, falling back to vectorized PyTorch.

    Returns a tuple of `(advantages, target_values)` where:
        - advantages: Advantage estimates masked beyond completion lengths.
        - target_values: TD(lambda) return targets (`advantages + values`) for critic
          training.
    """
    is_accelerator = (
        rewards.is_cuda
        or (hasattr(rewards, "is_mps") and rewards.is_mps)
        or rewards.device.type in ("cuda", "mps")
    )

    if TRITON_AVAILABLE and is_accelerator:
        return compute_gae_triton(
            rewards=rewards,
            values=values,
            next_values=next_values,
            dones=dones,
            seq_lens=seq_lens,
            gamma=gamma,
            lam=lam,
        )

    return compute_gae_torch(
        rewards=rewards,
        values=values,
        next_values=next_values,
        dones=dones,
        seq_lens=seq_lens,
        gamma=gamma,
        lam=lam,
    )
