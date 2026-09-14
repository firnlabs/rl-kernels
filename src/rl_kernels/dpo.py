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
    def _stable_logsigmoid(logits):
        abs_logits = tl.abs(logits)
        log_sig_abs = tl.log(1 + tl.exp(-abs_logits))
        return tl.where(logits >= 0, -log_sig_abs, logits - log_sig_abs)

    @triton.jit
    def dpo_loss_forward_kernel(
        policy_chosen_logps_ptr,
        policy_rejected_logps_ptr,
        ref_chosen_logps_ptr,
        ref_rejected_logps_ptr,
        losses_ptr,
        chosen_rewards_ptr,
        rejected_rewards_ptr,
        logits_ptr,
        n_elements,
        beta,
        label_smoothing: tl.constexpr,  # enable branch pruning
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        # load inputs
        policy_chosen_logps = tl.load(policy_chosen_logps_ptr + offsets, mask=mask)
        policy_rejected_logps = tl.load(policy_rejected_logps_ptr + offsets, mask=mask)
        ref_chosen_logps = tl.load(ref_chosen_logps_ptr + offsets, mask=mask)
        ref_rejected_logps = tl.load(ref_rejected_logps_ptr + offsets, mask=mask)

        chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
        rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)
        logits = chosen_rewards - rejected_rewards

        if label_smoothing == 0.0:
            losses = -_stable_logsigmoid(logits)  # ty: ignore[invalid-argument-type]
        else:
            losses = (
                -(1.0 - label_smoothing) * _stable_logsigmoid(logits)  # ty: ignore[invalid-argument-type]
                - label_smoothing * _stable_logsigmoid(-logits)  # ty: ignore[invalid-argument-type]
            )

        # store outputs
        tl.store(losses_ptr + offsets, losses, mask=mask)
        tl.store(chosen_rewards_ptr + offsets, chosen_rewards, mask=mask)
        tl.store(rejected_rewards_ptr + offsets, rejected_rewards, mask=mask)
        tl.store(logits_ptr + offsets, logits, mask=mask)

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
        losses = torch.empty_like(policy_chosen_logps)
        chosen_rewards = torch.empty_like(policy_chosen_logps)
        rejected_rewards = torch.empty_like(policy_chosen_logps)
        logits = torch.empty_like(policy_chosen_logps)

        n_elements = losses.numel()
        grid = lambda meta: (  # noqa: E731
            triton.cdiv(n_elements, meta["BLOCK_SIZE"]),  # ty: ignore[invalid-argument-type]
        )
        dpo_loss_forward_kernel[grid](
            policy_chosen_logps,
            policy_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
            losses,
            chosen_rewards,
            rejected_rewards,
            logits,
            n_elements,
            beta,
            label_smoothing,  # ty: ignore[invalid-argument-type]
            BLOCK_SIZE=1024,  # ty: ignore[invalid-argument-type]
        )

        return losses, chosen_rewards, rejected_rewards, logits


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
        ) - label_smoothing * F.logsigmoid(-logits)

    return losses, chosen_rewards, rejected_rewards, logits


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
        loss_fn = _compute_dpo_loss_triton
    else:
        loss_fn = _compute_dpo_loss_torch

    losses, chosen_rewards, rejected_rewards, logits = loss_fn(
        policy_chosen_logps=policy_chosen_logps,
        policy_rejected_logps=policy_rejected_logps,
        ref_chosen_logps=ref_chosen_logps,
        ref_rejected_logps=ref_rejected_logps,
        beta=beta,
        label_smoothing=label_smoothing,
    )

    return losses.mean(), chosen_rewards, rejected_rewards, logits
