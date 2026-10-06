"""Triton kernels for Direct Preference Optimization (DPO)."""

from typing import Any
from typing import Protocol

from beartype import beartype
from jaxtyping import Float
from jaxtyping import jaxtyped
import torch
from torch import Tensor
from torch.nn import functional as F

from rl_kernels.triton_helper import TRITON_AVAILABLE

if TRITON_AVAILABLE:  # pragma: no cover
    import triton  # ty: ignore[unresolved-import]  # noqa: F401
    import triton.language as tl  # ty: ignore[unresolved-import]  # noqa: F401

    @triton.jit
    def _stable_logsigmoid(logits):
        abs_logits = tl.abs(logits)
        log_sig_abs = tl.log(1 + tl.exp(-abs_logits))
        return tl.where(logits >= 0, -log_sig_abs, logits - log_sig_abs)

    @triton.jit
    def dpo_losses_forward_kernel(
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
    def dpo_loss_backward_kernel(
        logits_ptr,
        grad_policy_chosen_logps_ptr,
        grad_policy_rejected_logps_ptr,
        grad_output,
        n_elements,
        beta,
        label_smoothing,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        logits = tl.load(logits_ptr + offsets, mask=mask)
        sig = tl.sigmoid(logits)  # ty: ignore[invalid-argument-type]
        grad_policy_chosen_logps = (
            grad_output / n_elements * beta * (sig - (1.0 - label_smoothing))
        )
        grad_policy_rejected_logps = -grad_policy_chosen_logps

        tl.store(
            grad_policy_chosen_logps_ptr + offsets, grad_policy_chosen_logps, mask=mask
        )
        tl.store(
            grad_policy_rejected_logps_ptr + offsets,
            grad_policy_rejected_logps,
            mask=mask,
        )

    def _compute_dpo_losses_triton[B](
        *,
        policy_chosen_logps: Float[Tensor, "B"],
        policy_rejected_logps: Float[Tensor, "B"],
        ref_chosen_logps: Float[Tensor, "B"],
        ref_rejected_logps: Float[Tensor, "B"],
        beta: float = 0.1,
        label_smoothing: float = 0.0,
    ) -> tuple[
        Float[Tensor, "B"], Float[Tensor, "B"], Float[Tensor, "B"], Float[Tensor, "B"]
    ]:
        losses = torch.empty_like(policy_chosen_logps)
        chosen_rewards = torch.empty_like(policy_chosen_logps)
        rejected_rewards = torch.empty_like(policy_chosen_logps)
        logits = torch.empty_like(policy_chosen_logps)

        n_elements = losses.numel()
        grid = lambda meta: (  # noqa: E731
            triton.cdiv(n_elements, meta["BLOCK_SIZE"]),  # ty: ignore[invalid-argument-type]
        )
        dpo_losses_forward_kernel[grid](
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

    def _compute_dpo_loss_backward_triton[B](
        logits: Float[Tensor, "B"],
        grad_output: Float[Tensor, ""] | float = 1.0,
        beta: float = 0.1,
        label_smoothing: float = 0.0,
    ) -> tuple[Float[Tensor, "B"], Float[Tensor, "B"]]:
        grad_policy_chosen_logps = torch.empty_like(logits)
        grad_policy_rejected_logps = torch.empty_like(logits)
        n_elements = logits.numel()

        if isinstance(grad_output, Tensor):
            grad_output = float(grad_output.item())
        else:
            grad_output = float(grad_output)

        grid = lambda meta: (  # noqa: E731
            triton.cdiv(n_elements, meta["BLOCK_SIZE"]),  # ty: ignore[invalid-argument-type]
        )
        dpo_loss_backward_kernel[grid](
            logits,
            grad_policy_chosen_logps,
            grad_policy_rejected_logps,
            grad_output,
            n_elements,
            beta,
            label_smoothing,
            BLOCK_SIZE=1024,  # ty: ignore[invalid-argument-type]
        )
        return grad_policy_chosen_logps, grad_policy_rejected_logps


def _compute_dpo_losses_torch[B](
    *,
    policy_chosen_logps: Float[Tensor, "B"],
    policy_rejected_logps: Float[Tensor, "B"],
    ref_chosen_logps: Float[Tensor, "B"],
    ref_rejected_logps: Float[Tensor, "B"],
    beta: float = 0.1,
    label_smoothing: float = 0.0,
) -> tuple[
    Float[Tensor, "B"], Float[Tensor, "B"], Float[Tensor, "B"], Float[Tensor, "B"]
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

    return losses, chosen_rewards.detach(), rejected_rewards.detach(), logits.detach()


def _compute_dpo_loss_backward_torch[B](
    *,
    logits: Float[Tensor, "B"],
    grad_output: Float[Tensor, ""] | float = 1.0,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
) -> tuple[Float[Tensor, "B"], Float[Tensor, "B"]]:
    B = logits.shape[0]
    sig = F.sigmoid(logits)
    grad_policy_chosen_logps = grad_output / B * beta * (sig - (1.0 - label_smoothing))
    grad_plicy_rejected_logps = -grad_policy_chosen_logps
    return grad_policy_chosen_logps, grad_plicy_rejected_logps


class AutogradFunctionContext(Protocol):  # pragma: no cover
    saved_tensors: tuple[Tensor, ...]
    beta: float
    label_smoothing: float

    def save_for_backward(self, *tensors: Tensor) -> None: ...
    def mark_non_differentiable(self, *tensors: Tensor) -> None: ...


class TritonDPOLossFunction[B](torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: AutogradFunctionContext,
        policy_chosen_logps: Tensor,
        policy_rejected_logps: Tensor,
        ref_chosen_logps: Tensor,
        ref_rejected_logps: Tensor,
        beta: float = 0.1,
        label_smoothing: float = 0.0,
    ) -> tuple[
        Float[Tensor, ""], Float[Tensor, "B"], Float[Tensor, "B"], Float[Tensor, "B"]
    ]:
        loss_fn = (
            _compute_dpo_losses_triton
            if TRITON_AVAILABLE
            else _compute_dpo_losses_torch
        )

        losses, chosen_rewards, rejected_rewards, logits = loss_fn(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            ref_chosen_logps=ref_chosen_logps,
            ref_rejected_logps=ref_rejected_logps,
            beta=beta,
            label_smoothing=label_smoothing,
        )

        ctx.save_for_backward(logits)
        ctx.beta = beta
        ctx.label_smoothing = label_smoothing

        ctx.mark_non_differentiable(chosen_rewards, rejected_rewards, logits)

        loss = losses.mean()
        return loss, chosen_rewards, rejected_rewards, logits

    @staticmethod
    def backward(
        ctx: AutogradFunctionContext,
        grad_loss: Float[Tensor, ""],
        *grad_metrics: Any,  # unused
    ) -> tuple[
        Float[Tensor, "B"], Float[Tensor, "B"], None, None, None, None
    ]:  # pragma: no cover
        backward_fn = (
            _compute_dpo_loss_backward_triton
            if TRITON_AVAILABLE
            else _compute_dpo_loss_backward_torch
        )

        beta = ctx.beta
        label_smoothing = ctx.label_smoothing
        (logits,) = ctx.saved_tensors

        grad_policy_chosen_logps, grad_policy_rejected_logps = backward_fn(
            logits=logits,
            grad_output=grad_loss,
            beta=beta,
            label_smoothing=label_smoothing,
        )
        return (
            grad_policy_chosen_logps,
            grad_policy_rejected_logps,
            None,
            None,
            None,
            None,
        )


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
    loss, chosen_rewards, rejected_rewards, logits = TritonDPOLossFunction.apply(
        policy_chosen_logps=policy_chosen_logps,
        policy_rejected_logps=policy_rejected_logps,
        ref_chosen_logps=ref_chosen_logps,
        ref_rejected_logps=ref_rejected_logps,
        beta=beta,
        label_smoothing=label_smoothing,
    )

    return loss, chosen_rewards, rejected_rewards, logits
