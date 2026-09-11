"""Triton kernels for Direct Preference Optimization (DPO)."""

from beartype import beartype
from jaxtyping import Float
from jaxtyping import jaxtyped
import torch
from torch import Tensor
from torch.nn import functional as F

try:  # pragma: no cover
    import triton  # ty: ignore[unresolved-import]  # noqa: F401
    import triton.language as tl  # ty: ignore[unresolved-import]  # noqa: F401

    TRITON_AVAILABLE = torch.cuda.is_available()
    print("Triton is available")
except ImportError:
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:  # pragma: no cover

    @triton.jit
    def dpo_loss_forward_kernel():
        raise NotImplementedError("To be implemented")

    @triton.jit
    def dpo_loss_backward_kernel():
        raise NotImplementedError("To be implemented")

    def _compute_dpo_loss_triton[B](
        *,
        policy_chosen_logps: Float[Tensor, "B"],
        policy_rejected_logps: Float[Tensor, "B"],
        ref_chosen_logps: Float[Tensor, "B"],
        ref_rejected_logps: Float[Tensor, "B"],
        beta: float = 0.1,
        label_smoothing: float = 0.0,
    ) -> tuple[
        Float[Tensor, ""], Float[Tensor, "B"], Float[Tensor, "B"], Float[Tensor, "B"]
    ]:
        raise NotImplementedError("To be implemented")


def _compute_dpo_loss_torch[B](
    *,
    policy_chosen_logps: Float[Tensor, "B"],
    policy_rejected_logps: Float[Tensor, "B"],
    ref_chosen_logps: Float[Tensor, "B"],
    ref_rejected_logps: Float[Tensor, "B"],
    beta: float = 0.1,
    label_smoothing: float = 0.0,
) -> tuple[
    Float[Tensor, ""], Float[Tensor, "B"], Float[Tensor, "B"], Float[Tensor, "B"]
]:
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)
    logits = chosen_rewards - rejected_rewards

    if label_smoothing == 0.0:
        losses = -F.logsigmoid(logits)
    else:
        losses = -(1.0 - label_smoothing) * F.logsigmoid(
            logits
        ) - label_smoothing * F.logsigmoid(-logits)  # (B,)

    return losses.mean(), chosen_rewards, rejected_rewards, logits


@jaxtyped(typechecker=beartype)
def compute_dpo_loss[B](
    *,
    policy_chosen_logps: Float[Tensor, "B"],
    policy_rejected_logps: Float[Tensor, "B"],
    ref_chosen_logps: Float[Tensor, "B"],
    ref_rejected_logps: Float[Tensor, "B"],
    beta: float = 0.1,
    label_smoothing: float = 0.0,
) -> tuple[
    Float[Tensor, ""], Float[Tensor, "B"], Float[Tensor, "B"], Float[Tensor, "B"]
]:
    if TRITON_AVAILABLE:
        return _compute_dpo_loss_triton(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            ref_chosen_logps=ref_chosen_logps,
            ref_rejected_logps=ref_rejected_logps,
            beta=beta,
            label_smoothing=label_smoothing,
        )

    return _compute_dpo_loss_torch(
        policy_chosen_logps=policy_chosen_logps,
        policy_rejected_logps=policy_rejected_logps,
        ref_chosen_logps=ref_chosen_logps,
        ref_rejected_logps=ref_rejected_logps,
        beta=beta,
        label_smoothing=label_smoothing,
    )
