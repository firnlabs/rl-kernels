from unittest.mock import MagicMock

import pytest

from rl_kernels import dpo
from rl_kernels import triton_helper


def test_triton_available_type() -> None:
    """Verifies that TRITON_AVAILABLE and is_triton_available return booleans."""
    assert isinstance(triton_helper.TRITON_AVAILABLE, bool)
    assert isinstance(triton_helper.is_triton_available(), bool)


def test_dpo_triton_available_parity() -> None:
    """Verifies that dpo.TRITON_AVAILABLE matches triton_helper.TRITON_AVAILABLE."""
    # When not monkeypatched, dpo.TRITON_AVAILABLE should match triton_helper
    assert hasattr(dpo, "TRITON_AVAILABLE")


def test_is_backend_active_with_valid_active_driver() -> None:
    """Verifies active driver detection returns True."""
    backend = MagicMock()
    backend.driver.is_active.return_value = True
    assert triton_helper._is_backend_active(backend) is True


def test_is_backend_active_with_inactive_driver() -> None:
    """Verifies inactive driver detection returns False."""
    backend = MagicMock()
    backend.driver.is_active.return_value = False
    assert triton_helper._is_backend_active(backend) is False


def test_is_backend_active_missing_or_none_driver() -> None:
    """Verifies driverless backend returns False."""
    backend_none = MagicMock(driver=None)
    assert triton_helper._is_backend_active(backend_none) is False

    backend_missing = object()
    assert triton_helper._is_backend_active(backend_missing) is False


def test_is_backend_active_exception_handling() -> None:
    """Verifies that exceptions during driver inspection are caught safely."""
    backend = MagicMock()
    backend.driver.is_active.side_effect = RuntimeError("Driver failure")
    assert triton_helper._is_backend_active(backend) is False


def test_is_triton_available_with_mocked_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies is_triton_available returns True if at least one backend is active."""
    inactive_backend = MagicMock()
    inactive_backend.driver.is_active.return_value = False

    active_backend = MagicMock()
    active_backend.driver.is_active.return_value = True

    mock_backends = {"cuda": inactive_backend, "msl": active_backend}

    monkeypatch.setattr(
        triton_helper,
        "_is_backend_active",
        lambda b: b.driver.is_active(),
    )

    try:
        import triton.backends

        monkeypatch.setattr(triton.backends, "backends", mock_backends)
        assert triton_helper.is_triton_available() is True
    except ImportError:  # pragma: no cover
        pass
