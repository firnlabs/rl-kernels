"""Tests for dpo.py."""

import pytest
import torch

from rl_kernels import dpo


@pytest.fixture(autouse=True)
def disable_triton(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(dpo, "TRITON_AVAILABLE", False)


@pytest.mark.parametrize("label_smoothing", [0.0, 0.05])
def test_compute_dpo_loss(label_smoothing: float):
    batch_size = 256
    policy_chosen_logps = torch.randn(batch_size)
    policy_rejected_logps = torch.randn(batch_size)
    ref_chosen_logps = torch.randn(batch_size)
    ref_rejected_logps = torch.randn(batch_size)
    beta = 0.1

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
def test_compute_dpo_loss_triton_torch_parity(label_smoothing: float):
    device = "cuda"
    batch_size = 256
    policy_chosen_logps = torch.randn(batch_size).to(device)
    policy_rejected_logps = torch.randn(batch_size).to(device)
    ref_chosen_logps = torch.randn(batch_size).to(device)
    ref_rejected_logps = torch.randn(batch_size).to(device)
    beta = 0.1

    ref_losses, ref_chosen_rewards, ref_rejected_rewards, ref_logits = (
        dpo._compute_dpo_loss_torch(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            ref_chosen_logps=ref_chosen_logps,
            ref_rejected_logps=ref_rejected_logps,
            beta=beta,
            label_smoothing=label_smoothing,
        )
    )

    losses, chosen_rewards, rejected_rewards, logits = dpo._compute_dpo_loss_triton(
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
