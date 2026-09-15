"""Helper utilities for OpenAI Triton backend detection and management."""

from typing import Any

__all__ = ["TRITON_AVAILABLE", "is_triton_available"]


def _is_backend_active(backend: Any) -> bool:
    """Checks if a Triton backend driver is active safely.

    Args:
        backend: A Triton backend instance.

    Returns:
        True if the backend driver is active, False otherwise.
    """
    try:
        driver = getattr(backend, "driver", None)
        if driver is not None and hasattr(driver, "is_active"):
            return bool(driver.is_active())
    except Exception:
        pass
    return False


def is_triton_available() -> bool:
    """Determines whether OpenAI Triton is installed and has an active hardware driver.

    Supported drivers:
    - Apple Silicon: triton-msl + Xcode Metal toolchain
    - NVIDIA: CUDA driver
    - AMD: ROCm/HIP driver

    Returns:
        True if Triton is installed and a hardware driver is active, False otherwise.
    """
    try:  # pragma: no cover
        import triton  # ty: ignore[unresolved-import]  # noqa: F401
        from triton.backends import backends  # ty: ignore[unresolved-import]
        import triton.language as tl  # ty: ignore[unresolved-import]  # noqa: F401

        # True only if a supported hardware driver is compiled and active:
        # - Apple Silicon: triton-msl + Xcode Metal toolchain
        # - NVIDIA: CUDA driver
        # - AMD: ROCm/HIP driver
        return any(_is_backend_active(b) for b in backends.values())
    except Exception:  # pragma: no cover
        return False


TRITON_AVAILABLE: bool = is_triton_available()
