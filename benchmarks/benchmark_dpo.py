"""Benchmarking script for DPO Triton kernel against PyTorch baseline."""

import os
import sys

import torch
import triton
import triton.testing
import typer

from rl_kernels import dpo
from rl_kernels.triton_helper import TRITON_AVAILABLE

# Ensure benchmark runs on real hardware rather than Triton interpreter
os.environ.pop("TRITON_INTERPRET", None)

# Fix for headless environments (e.g. Colab / Jupyter) where MPLBACKEND is set to
# matplotlib_inline, which is not installed in the isolated virtualenv
if "matplotlib_inline" in os.environ.get("MPLBACKEND", ""):
    os.environ["MPLBACKEND"] = "Agg"

app = typer.Typer(
    name="benchmark-dpo",
    help="Benchmark DPO Triton kernel vs PyTorch baseline.",
    add_completion=False,
)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[2**i for i in range(10, 20)],
        line_arg="provider",
        line_vals=["triton", "pytorch"],
        line_names=["Triton", "PyTorch"],
        styles=[("blue", "-"), ("green", "--")],
        ylabel="GB/s",
        plot_name="dpo-loss-performance",
        args={},
    )
)
def benchmark(N: int, provider: str) -> tuple[float, float, float]:
    """Runs a single benchmark trial for a given size and provider.

    Args:
        N: Sequence/batch length (number of elements).
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
    policy_chosen_logps = torch.randn(N, requires_grad=True, device=device)
    policy_rejected_logps = torch.randn(N, requires_grad=True, device=device)
    ref_chosen_logps = torch.randn(N, requires_grad=False, device=device)
    ref_rejected_logps = torch.randn(N, requires_grad=False, device=device)
    beta = 0.1
    label_smoothing = 0.05

    quantiles = [0.5, 0.2, 0.8]

    if provider == "pytorch":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: dpo._compute_dpo_losses_torch(
                policy_chosen_logps=policy_chosen_logps,
                policy_rejected_logps=policy_rejected_logps,
                ref_chosen_logps=ref_chosen_logps,
                ref_rejected_logps=ref_rejected_logps,
                beta=beta,
                label_smoothing=label_smoothing,
            ),
            quantiles=quantiles,
        )
    elif provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: dpo._compute_dpo_losses_triton(
                policy_chosen_logps=policy_chosen_logps,
                policy_rejected_logps=policy_rejected_logps,
                ref_chosen_logps=ref_chosen_logps,
                ref_rejected_logps=ref_rejected_logps,
                beta=beta,
                label_smoothing=label_smoothing,
            ),
            quantiles=quantiles,
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")

    def get_gbps(latency_ms: float) -> float:
        return 4 * policy_chosen_logps.element_size() * N / latency_ms * 1e-6

    return get_gbps(ms), get_gbps(max_ms), get_gbps(min_ms)


def run_benchmark(
    *,
    show_plots: bool = False,
    print_data: bool = True,
    save_path: str = "benchmarks/results",
) -> None:
    """Executes the DPO Triton vs PyTorch benchmark.

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
        help="Directory to save benchmark plots and output files.",
    ),
    no_print: bool = typer.Option(
        False,
        "--no-print",
        help="Suppress printing results data table to console.",
    ),
) -> None:
    """Benchmark DPO Triton kernel against PyTorch baseline.

    Args:
        show_plots: Display generated benchmark plots interactively.
        save_path: Directory to save benchmark plots and output files.
        no_print: Suppress printing results data table to console.
    """
    run_benchmark(
        show_plots=show_plots,
        print_data=not no_print,
        save_path=save_path,
    )


if __name__ == "__main__":
    app()
