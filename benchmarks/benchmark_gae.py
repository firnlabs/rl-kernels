"""Benchmarking script for GAE Triton kernel against PyTorch baseline."""

import os
import sys

import torch
import triton
import triton.testing
import typer

from rl_kernels import gae
from rl_kernels.triton_helper import TRITON_AVAILABLE

# Ensure benchmark runs on real hardware rather than Triton interpreter
os.environ.pop("TRITON_INTERPRET", None)

# Fix for headless environments (e.g. Colab / Jupyter) where MPLBACKEND is set to
# matplotlib_inline, which is not installed in the isolated virtualenv
if "matplotlib_inline" in os.environ.get("MPLBACKEND", ""):
    os.environ["MPLBACKEND"] = "Agg"

app = typer.Typer(
    name="benchmark-gae",
    help="Benchmark GAE Triton kernel vs PyTorch baseline.",
    add_completion=False,
)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["T"],
        x_vals=[2**i for i in range(6, 14)],  # 64 to 8192 sequence length
        line_arg="provider",
        line_vals=["triton", "pytorch"],
        line_names=["Triton", "PyTorch"],
        styles=[("blue", "-"), ("green", "--")],
        ylabel="GB/s",
        plot_name="gae-performance",
        args={"B": 16},
    )
)
def benchmark(B: int, T: int, provider: str) -> tuple[float, float, float]:
    """Runs a single benchmark trial for a given size and provider.

    Args:
        B: Batch size.
        T: Sequence length.
        provider: Implementation provider ('triton' or 'pytorch').

    Returns:
        A tuple of (median_gbps, max_gbps, min_gbps).

    Raises:
        RuntimeError: If OpenAI Triton is not available.
        ValueError: If an unknown provider is specified.
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError("Requires OpenAI Triton availability.")

    device = torch.accelerator.current_accelerator()
    rewards = torch.randn((B, T), device=device, dtype=torch.float32)
    values = torch.randn((B, T), device=device, dtype=torch.float32)
    next_values = torch.randn((B,), device=device, dtype=torch.float32)
    dones = (torch.rand((B, T), device=device) > 0.95).float()
    seq_lens = torch.full((B,), T, device=device, dtype=torch.int64)
    gamma = 0.99
    lam = 0.95

    quantiles = [0.5, 0.2, 0.8]

    if provider == "pytorch":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: gae.compute_gae_torch(
                rewards=rewards,
                values=values,
                next_values=next_values,
                dones=dones,
                seq_lens=seq_lens,
                gamma=gamma,
                lam=lam,
            ),
            quantiles=quantiles,
        )
    elif provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: gae.compute_gae_triton(
                rewards=rewards,
                values=values,
                next_values=next_values,
                dones=dones,
                seq_lens=seq_lens,
                gamma=gamma,
                lam=lam,
            ),
            quantiles=quantiles,
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")

    def get_gbps(latency_ms: float) -> float:
        # GAE transfers roughly:
        # Read: rewards (BT), values (BT), dones (BT)
        # Write: advantages (BT), target_values (BT)
        # Total ≈ 5 tensors * B * T * 4 bytes
        total_bytes = 5 * B * T * rewards.element_size()
        return total_bytes / (latency_ms * 1e-3) / 1e9

    return get_gbps(ms), get_gbps(max_ms), get_gbps(min_ms)


def run_benchmark(
    *,
    show_plots: bool = False,
    print_data: bool = True,
    save_path: str = "benchmarks/results",
) -> None:
    """Executes the GAE Triton vs PyTorch benchmark.

    Args:
        show_plots: Whether to display performance plots interactively.
        print_data: Whether to print results to stdout.
        save_path: Directory path where output plots and data are saved.
    """
    if not TRITON_AVAILABLE:
        print(
            "Triton is not available with an active hardware driver. "
            "Skipping benchmark.",
            file=sys.stderr,
        )
        return

    benchmark.run(
        show_plots=show_plots,
        print_data=print_data,
        save_path=save_path,
    )


@app.command()
def main(
    show_plots: bool = typer.Option(
        False,
        "--show-plots",
        help="Display generated benchmark plots interactively.",
    ),
    save_path: str = typer.Option(
        "benchmarks/results",
        "--save-path",
        "-s",
        help="Directory path to save benchmark charts and data.",
    ),
) -> None:
    """CLI entry point for GAE performance benchmarking."""
    run_benchmark(show_plots=show_plots, save_path=save_path)


if __name__ == "__main__":
    app()
