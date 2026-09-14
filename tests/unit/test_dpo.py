"""Tests for dpo.py."""

from jaxtyping import Float
import pytest
import torch
from torch import Tensor

from rl_kernels import dpo


@pytest.fixture(autouse=True)
def disable_triton(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dpo, "TRITON_AVAILABLE", False)


@pytest.fixture
def batch_size() -> int:
    return 256


@pytest.fixture
def policy_chosen_logps[B](batch_size: int) -> Float[Tensor, "B"]:
    return torch.randn(batch_size)


@pytest.fixture
def policy_rejected_logps[B](batch_size: int) -> Float[Tensor, "B"]:
    return torch.randn(batch_size)


@pytest.fixture
def ref_chosen_logps[B](batch_size: int) -> Float[Tensor, "B"]:
    return torch.randn(batch_size)


@pytest.fixture
def ref_rejected_logps[B](batch_size: int) -> Float[Tensor, "B"]:
    return torch.randn(batch_size)


@pytest.fixture
def beta() -> float:
    return 0.1


@pytest.mark.parametrize("label_smoothing", [0.0, 0.05])
def test_compute_dpo_loss[B](
    batch_size: int,
    policy_chosen_logps: Float[Tensor, "B"],
    policy_rejected_logps: Float[Tensor, "B"],
    ref_chosen_logps: Float[Tensor, "B"],
    ref_rejected_logps: Float[Tensor, "B"],
    beta: float,
    label_smoothing: float,
) -> None:
    loss, chosen_rewards, rejected_rewards, logits = dpo.compute_dpo_loss(
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


@pytest.mark.skipif(
    not dpo.TRITON_AVAILABLE, reason="Requires OpenAI Triton availability"
)
@pytest.mark.parametrize("label_smoothing", [0.0, 0.05])
def test_compute_dpo_loss_triton_torch_parity[B](
    policy_chosen_logps: Float[Tensor, "B"],
    policy_rejected_logps: Float[Tensor, "B"],
    ref_chosen_logps: Float[Tensor, "B"],
    ref_rejected_logps: Float[Tensor, "B"],
    beta: float,
    label_smoothing: float,
) -> None:
    device = "cuda"
    policy_chosen_logps = policy_chosen_logps.to(device)
    policy_rejected_logps = policy_rejected_logps.to(device)
    ref_chosen_logps = ref_chosen_logps.to(device)
    ref_rejected_logps = ref_rejected_logps.to(device)

    ref_losses, ref_chosen_rewards, ref_rejected_rewards, ref_logits = (
        dpo._compute_dpo_losses_torch(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            ref_chosen_logps=ref_chosen_logps,
            ref_rejected_logps=ref_rejected_logps,
            beta=beta,
            label_smoothing=label_smoothing,
        )
    )

    losses, chosen_rewards, rejected_rewards, logits = dpo._compute_dpo_losses_triton(
        policy_chosen_logps=policy_chosen_logps,
        policy_rejected_logps=policy_rejected_logps,
        ref_chosen_logps=ref_chosen_logps,
        ref_rejected_logps=ref_rejected_logps,
        beta=beta,
        label_smoothing=label_smoothing,
    )

    torch.testing.assert_close(losses, ref_losses, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(chosen_rewards, ref_chosen_rewards, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(
        rejected_rewards, ref_rejected_rewards, rtol=1e-5, atol=1e-5
    )
    torch.testing.assert_close(logits, ref_logits, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("label_smoothing", [0.0, 0.05])
def test_dpo_loss_backward_torch_grad_check[B](
    policy_chosen_logps: Float[Tensor, "B"],
    policy_rejected_logps: Float[Tensor, "B"],
    ref_chosen_logps: Float[Tensor, "B"],
    ref_rejected_logps: Float[Tensor, "B"],
    beta: float,
    label_smoothing: float,
) -> None:
    policy_chosen_logps = policy_chosen_logps.clone().detach().requires_grad_(True)
    policy_rejected_logps = policy_rejected_logps.clone().detach().requires_grad_(True)

    losses, _, _, logits = dpo._compute_dpo_losses_torch(
        policy_chosen_logps=policy_chosen_logps,
        policy_rejected_logps=policy_rejected_logps,
        ref_chosen_logps=ref_chosen_logps,
        ref_rejected_logps=ref_rejected_logps,
        beta=beta,
        label_smoothing=label_smoothing,
    )
    loss = losses.mean()

    grad_policy_chosen_expected, grad_policy_rejected_expected = torch.autograd.grad(
        loss, (policy_chosen_logps, policy_rejected_logps)
    )

    grad_policy_chosen, grad_policy_rejected = dpo._compute_dpo_loss_backward_torch(
        logits=logits.detach(),
        beta=beta,
        label_smoothing=label_smoothing,
    )

    torch.testing.assert_close(
        grad_policy_chosen, grad_policy_chosen_expected, rtol=1e-5, atol=1e-5
    )
    torch.testing.assert_close(
        grad_policy_rejected, grad_policy_rejected_expected, rtol=1e-5, atol=1e-5
    )


@pytest.mark.skipif(
    not dpo.TRITON_AVAILABLE, reason="Requires OpenAI Triton availability"
)
@pytest.mark.parametrize("label_smoothing", [0.0, 0.05])
def test_dpo_loss_backward_triton_torch_parity[B](
    policy_chosen_logps: Float[Tensor, "B"],
    policy_rejected_logps: Float[Tensor, "B"],
    ref_chosen_logps: Float[Tensor, "B"],
    ref_rejected_logps: Float[Tensor, "B"],
    beta: float,
    label_smoothing: float,
) -> None:
    device = "cuda"
    policy_chosen_logps = policy_chosen_logps.to(device)
    policy_rejected_logps = policy_rejected_logps.to(device)
    ref_chosen_logps = ref_chosen_logps.to(device)
    ref_rejected_logps = ref_rejected_logps.to(device)
    # alternate way to compute dpo logits
    policy_ratio = policy_chosen_logps - policy_rejected_logps
    ref_ratio = ref_chosen_logps - ref_rejected_logps
    logits = beta * (policy_ratio - ref_ratio)
    grad_output = 1.0

    ref_grad_chosen, ref_grad_rejected = dpo._compute_dpo_loss_backward_torch(
        logits=logits,
        grad_output=grad_output,
        beta=beta,
        label_smoothing=label_smoothing,
    )

    grad_chosen, grad_rejected = dpo._compute_dpo_loss_backward_triton(
        logits=logits,
        grad_output=grad_output,
        beta=beta,
        label_smoothing=label_smoothing,
    )

    torch.testing.assert_close(grad_chosen, ref_grad_chosen, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(grad_rejected, ref_grad_rejected, rtol=1e-5, atol=1e-5)
