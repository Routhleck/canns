"""Factory + pre-compute helpers for :mod:`.backends`.

:func:`make_irec_backend` is the single dispatch point that turns a
``(family, mode, dim, length, k, target_err_mrad)`` request into a
concrete :class:`IrecBackend` from :mod:`._backends`. The SVD and
FFT pre-compute helpers (:func:`_lowrank_factors`,
:func:`_fft_kernel_1d`, :func:`_fft_kernel_2d`) live here too —
they're internal to the factory and not part of the public surface.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np

from ._backends import (
    CirculantFFTIrec1D,
    CirculantFFTIrec2D,
    DenseIrec,
    LowRankIrec,
)
from ._protocol import IrecBackend  # noqa: F401  (re-exported for type hints)
from ._spectrum import _pick_k_for_err_target

# ---------------------------------------------------------------------------
# Pre-compute helpers (called by the factory)
# ---------------------------------------------------------------------------


def _lowrank_factors(conn_mat, k: int) -> tuple[jax.Array, jax.Array, int]:
    """Compute the balanced rank-``k`` SVD factors of ``conn_mat``.

    Returns ``(U_l, V_l, k_eff)`` where ``k_eff = min(k, n)`` is the
    rank actually used (clamped to the matrix size when the user
    asks for more components than the matrix has).
    """
    U, S, Vt = np.linalg.svd(np.asarray(conn_mat), full_matrices=False)
    k_eff = min(k, S.shape[0])
    sqrtS = np.sqrt(S[:k_eff].astype(np.float32))
    U_l = jax.numpy.asarray(U[:, :k_eff].astype(np.float32) * sqrtS)
    V_l = jax.numpy.asarray(Vt[:k_eff, :].T.astype(np.float32) * sqrtS)
    return U_l, V_l, k_eff


def _fft_kernel_1d(conn_mat) -> jax.Array | None:
    """Return the 1D FFT of the circulant kernel, or ``None`` if the
    grid is not circulant (i.e. ``endpoint=True`` in the canns default).

    Detection: a right-circulant matrix has ``row[0] == row[-1]``. On
    the canns ``endpoint=True`` grid, ``row[0][0] = row[-1][0]`` (both
    evaluate the kernel at distance 0), but ``row[0][1] != row[-1][1]``
    (the wrap brings the two endpoints to the same place while the
    non-endpoint points are spaced differently), so the test fires
    correctly on the default grid and not on ``endpoint=False``.
    """
    first = np.asarray(conn_mat[0, :], dtype=np.float32)
    n = first.shape[0]
    if n > 1 and np.isclose(first[0], first[-1], rtol=1e-7, atol=1e-7):
        return None
    return jax.numpy.asarray(jnp.fft.fft(jnp.asarray(first)))


def _fft_kernel_2d(conn_mat, length: int) -> jax.Array | None:
    """Return the 2D FFT of the doubly-circulant kernel, or ``None`` if
    the grid is not circulant.

    Detection: a doubly-circulant matrix has its first row equal to
    a left/right circular shift of itself. On the canns
    ``endpoint=True`` default, the four corners of the ``L × L`` first
    row are all ``f(0, 0) = max`` (because of the wrap), so
    ``K[0, 0] ≈ K[0, L-1]``; on ``endpoint=False`` they differ.
    """
    K_first = np.asarray(conn_mat[0, :], dtype=np.float32).reshape(length, length)
    if length > 1 and np.isclose(K_first[0, 0], K_first[0, length - 1], rtol=1e-7, atol=1e-7):
        return None
    return jax.numpy.asarray(jnp.fft.fft2(jnp.asarray(K_first)))


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------


def make_irec_backend(
    model,
    *,
    family: str,
    mode: str,
    dim: int,
    length: int | None = None,
    k: int | None = None,
    target_err_mrad: float = 5.0,
) -> IrecBackend:
    """Build the ``Irec`` backend for ``model``'s recurrent matvec.

    Args:
        model: the model instance; consulted for ``conn_mat`` to drive
            the SVD or FFT pre-compute.
        family: ``"CANN1D"`` or ``"CANN2D"``; used to look up the
            default rank in :data:`ACCL_MODES`' ``ACCL_DEFAULT_K`` for
            ``"fast"``/``"ultra-fast"``.
        mode: one of :data:`ACCL_MODES`.
        dim: ``1`` or ``2``; selects the FFT kernel shape.
        length: only used when ``dim == 2`` (one side of the square grid).
        k: explicit low-rank truncation; overrides the mode default and
            the ``auto``-mode spectrum pick. Ignored for ``"normal"`` and
            ``"fft"``.
        target_err_mrad: only used when ``mode == "auto"`` and
            ``k is None``. If no lowrank truncation satisfies the
            budget (:func:`_pick_k_for_err_target` returns ``-1``),
            falls back to :class:`DenseIrec`.

    Returns:
        An :class:`IrecBackend` carrying its own pre-computed factors.
        The caller assigns it to ``self.irec_backend``.

    Raises:
        ValueError: if ``mode`` is not in :data:`ACCL_MODES`, or if
            ``k`` is provided but is not a positive int.
    """
    # Local import to avoid a circular import at module load
    # (accel/__init__.py defines ACCL_MODES; factory reads it).
    from . import ACCL_DEFAULT_K, ACCL_MODES

    if mode not in ACCL_MODES:
        raise ValueError(f"accl_mode must be one of {ACCL_MODES!r}, got {mode!r}")

    if mode == "normal":
        return DenseIrec(model.conn_mat)

    if mode == "fft":
        if dim == 1:
            K = _fft_kernel_1d(model.conn_mat)
        else:
            assert length is not None, "length is required for dim=2 FFT"
            K = _fft_kernel_2d(model.conn_mat, length)
        if K is None:
            # canns default ``endpoint=True`` grid: the wrap convention
            # duplicates the endpoints, which breaks the (doubly-)
            # circulant structure that the FFT formula relies on.
            # Fall back to dense with a warning so the model still works.
            warnings.warn(
                "accl_mode='fft' requested but the grid appears to "
                "use endpoint=True (x[0] = x[-1] on the torus), which "
                "breaks the circulant structure. Falling back to "
                "dense. To use FFT acceleration, set a custom grid "
                "with endpoint=False, e.g. "
                "`model.x = bm.linspace(-bm.pi, bm.pi, num, endpoint=False)`.",
                stacklevel=3,
            )
            return DenseIrec(model.conn_mat)
        if dim == 1:
            return CirculantFFTIrec1D(K)
        return CirculantFFTIrec2D(K, length)

    # Low-rank family: "fast" / "ultra-fast" / "auto"
    if mode == "auto" and k is None:
        S = np.linalg.svd(np.asarray(model.conn_mat), compute_uv=False)
        picked = _pick_k_for_err_target(S, target_err_mrad)
        if picked == -1:
            # Budget too tight for any lowrank; use dense.
            return DenseIrec(model.conn_mat)
        k = picked

    if k is None:
        # Mode is "fast" or "ultra-fast" (the "auto" branch above
        # always sets k, and "normal" / "fft" were handled earlier).
        k = ACCL_DEFAULT_K[(family, mode)]

    if not (isinstance(k, int) and k >= 1):
        raise ValueError(f"accl_k must be a positive int, got {k!r}")

    U_l, V_l, k = _lowrank_factors(model.conn_mat, k)
    return LowRankIrec(U_l, V_l, k=k, mode=mode)


__all__ = [
    "make_irec_backend",
]
