"""OpenAI Triton Vector Addition Smoke Test Kernel.

This script validates that the runtime environment has properly configured:
1. GPU drivers and accelerator device support (CUDA or Metal).
2. PyTorch with accelerator support.
3. OpenAI Triton JIT compilation pipeline (@triton.jit).
4. Memory boundary masking logic.
"""

import sys

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from rl_kernels.triton_helper import TRITON_AVAILABLE

if TRITON_AVAILABLE:
    import triton  # ty: ignore[unresolved-import]
    import triton.language as tl  # ty: ignore[unresolved-import]

    @triton.jit
    def vector_add_kernel(  # noqa: DOC108
        x_ptr,
        y_ptr,
        out_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Triton kernel for elementwise vector addition: out = x + y.

        Args:
            x_ptr: Pointer to first input vector in GPU memory.
            y_ptr: Pointer to second input vector in GPU memory.
            out_ptr: Pointer to output vector in GPU memory.
            n_elements: Total number of scalar elements.
            BLOCK_SIZE: Number of elements processed per thread block.
        """
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y
        tl.store(out_ptr + offsets, output, mask=mask)

    def triton_vector_add(
        x: torch.Tensor, y: torch.Tensor, block_size: int = 1024
    ) -> torch.Tensor:
        """Wrapper to invoke Triton vector addition kernel."""
        assert (
            x.is_cuda
            or (hasattr(x, "is_mps") and x.is_mps)
            or x.device.type
            in (
                "cuda",
                "mps",
            )
        ), "Tensor must be on an accelerator device."
        assert x.shape == y.shape, "Tensor shapes must match."
        assert x.dtype == y.dtype == torch.float32, "Only float32 supported."

        output = torch.empty_like(x)
        n_elements = output.numel()

        def grid(meta):
            return (
                triton.cdiv(n_elements, meta["BLOCK_SIZE"]),  # ty: ignore[invalid-argument-type]
            )

        vector_add_kernel[grid](
            x,
            y,
            output,
            n_elements,
            BLOCK_SIZE=block_size,  # ty: ignore[invalid-argument-type]
        )
        return output


def run_smoke_test() -> bool:
    """Runs diagnostics to test Triton availability and kernel execution."""
    print("=" * 60)
    print("OpenAI Triton Smoke Test Diagnostics")
    print("=" * 60)
    print(f"[*] Python:             {sys.version.split()[0]}")

    if not TORCH_AVAILABLE:
        print("❌ PyTorch is not installed.")
        return False

    print(f"[*] PyTorch Version:    {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    print(f"[*] CUDA Available:     {cuda_available}")
    print(f"[*] MPS Available:      {mps_available}")
    print(f"[*] Triton Available:   {TRITON_AVAILABLE}")

    if not TRITON_AVAILABLE:
        print("❌ OpenAI Triton is not available in the current environment.")
        return False

    if not (cuda_available or mps_available):
        print("⚠️ No GPU accelerator detected. Skipping kernel launch.")
        return False

    device = torch.device("cuda" if cuda_available else "mps")
    print(f"[*] Testing kernel launch on {device}...")

    n_elements = 4096
    x = torch.ones(n_elements, dtype=torch.float32, device=device)
    y = torch.full((n_elements,), 2.0, dtype=torch.float32, device=device)

    try:
        out = triton_vector_add(x, y, block_size=1024)
        expected = x + y
        assert torch.allclose(out, expected), "Triton output does not match expected."
        print("✅ Triton vector_add kernel succeeded and verified against PyTorch!")
        return True
    except Exception as e:
        print(f"❌ Kernel execution failed: {e}")
        return False


if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
