"""Triton kernels for Direct Preference Optimization (DPO)."""

from beartype import beartype
from jaxtyping import Float
from jaxtyping import jaxtyped
from torch import Tensor
from torch.nn import functional as F

try:
    import triton  # ty: ignore[unresolved-import]  # noqa: F401
    import triton.language as tl  # ty: ignore[unresolved-import]  #noqa: F401

    TRITON_AVAILABLE = True
    print("Triton is available")
except ImportError:
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:

    def dpo_loss_forward_kernel():
        pass

    def dpo_loss_backward_kernel():
        pass

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


if __name__ == "__main___":
    import torch

    def test_compute_dpo_loss_torch():
        batch_size = 256
        policy_chosen_logps = torch.randn(batch_size)
        policy_rejected_logps = torch.randn(batch_size)
        ref_chosen_logps = torch.randn(batch_size)
        ref_rejected_logps = torch.randn(batch_size)
        beta = 0.1
        label_smoothing = 0.05

        loss, chosen_rewards, rejected_rewards, logits = _compute_dpo_loss_torch(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            ref_chosen_logps=ref_chosen_logps,
            ref_rejected_logps=ref_rejected_logps,
            beta=beta,
            label_smoothing=label_smoothing,
        )

        assert loss.shape == ()
        assert chosen_rewards.shape == (batch_size,)
        assert rejected_rewards.shape == (batch_size,)
        assert logits.shape == (batch_size,)
