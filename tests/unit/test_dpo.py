"""Tests for dpo.py."""

import pytest
import torch

from rl_kernels import dpo


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
