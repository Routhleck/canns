"""Backends for the CANN recurrent matvec ``Irec = W @ r``.

Strategy pattern: each backend is a self-contained callable that
carries its own pre-computed factors and exposes a single
``__call__(r) -> Irec`` entry point.

``canns.models.basic.cann.BaseCANN1D/2D`` hold one backend on
``self.irec_backend`` and just call it from ``update()``. Adding a
new acceleration mode (e.g. block-circulant, Chebyshev, random
projection) means adding a new class here and dispatching on it from
:func:`make_irec_backend` — ``cann.py`` is not touched.

For backward compatibility the public symbols
:data:`ACCL_MODES`, :data:`ACCL_DEFAULT_K`, and
:func:`_pick_k_for_err_target` are re-exported from
``canns.models.basic.cann``.

Which backend should I use?
---------------------------

The five modes below trade off **accuracy** (how close the matvec
is to the full dense ``conn_mat @ r``) against **speed** (how many
FLOPs per step). The same CANN can be moved between modes at
runtime via ``model.set_accl_mode(...)`` without re-instantiating.

================== ========================================== ============ ============ ==================================
``accl_mode``       Math                                       Exact?       n=2048 speed  When to use
================== ========================================== ============ ============ ==================================
``"normal"``        ``W @ r``                                  Yes          1×            Default. Use for small ``num`` (≤ 128) where the overhead of any acceleration exceeds the savings, and as the reference for accuracy comparisons.
``"fast"``          ``U @ (Vᵀ @ r)`` from rank-``k`` SVD       No (k-tail)  30–245×       Recommended for production. ~5 mrad bump-position error at the recommended ranks (``CANN1D`` k=8, ``CANN2D`` k=32). See ``benchmarks/cann_lowrank``.
``"ultra-fast"``    ``U @ (Vᵀ @ r)`` at k=1 (or 4 for 2D)      No (coarser) 100–500×      Only when the connection spectrum decays extremely fast (narrow Gaussian ``a``) or when only the bump's existence matters, not its precise position.
``"auto"``          rank picked from SVD spectrum              No (≤ budget) varies        Pick a rank automatically to satisfy ``accl_target_err_mrad`` (default 5 mrad). The user-facing knob is the error budget, not the rank.
``"fft"``           ``real(ifft(K_fft ⊙ fft(r)))``             **Yes**      25–50×        **Only** on a clean ring/torus with ``endpoint=False``. The canns default ``endpoint=True`` grid is not circulant, so this mode falls back to dense with a warning. See the table below for setup.
================== ========================================== ============ ============ ==================================

Setup for the ``"fft"`` mode:

.. code-block:: python

    # canns default: x = bm.linspace(-pi, pi, num, endpoint=True)
    # → not circulant under canns's wrap convention. FFT mode will fall back.
    m = CANN1D(num=512, accl_mode="fft")  # silently becomes "normal"

    # To actually get the FFT speedup, override the grid:
    m = CANN1D(num=512, accl_mode="normal")
    m.x = bm.linspace(-bm.pi, bm.pi, 512, endpoint=False)
    m.conn_mat = m.make_conn()       # rebuild on the new grid
    m.set_accl_mode("fft")           # now the FFT kernel is exact

How to add a new backend
------------------------

Three steps, all in this file:

1. Write a class that holds whatever pre-computed factors you need
   and implements ``__call__(self, r)``. Mirror the existing
   ``DenseIrec`` / ``LowRankIrec`` / ``CirculantFFTIrec*`` shape.
2. Add a new value to :data:`ACCL_MODES` (and to
   :data:`ACCL_DEFAULT_K` if your mode has a per-family default rank).
3. Add a dispatch branch in :func:`make_irec_backend`.

``cann.py`` does not need to change. The model only sees a callable
that takes ``r`` and returns ``Irec``.
"""

from __future__ import annotations

import warnings
from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

#: Recognised values for ``accl_mode``. ``"normal"`` is full-rank; the
#: others are approximations of the recurrent matvec.
#: ``"auto"`` selects the rank dynamically from the SVD spectrum
#: to satisfy the ``accl_target_err_mrad`` budget; ``"fft"`` exploits
#: the circulant structure of the Gaussian distance kernel on a
#: uniform ring/torus and uses FFT to compute the matvec exactly
#: in O(n log n).
ACCL_MODES: tuple[str, ...] = ("normal", "fast", "ultra-fast", "auto", "fft")

#: Default rank for each ``(model family, mode)`` combination.
#:
#: The "fast" defaults are the values recommended by
#: ``benchmarks/cann_lowrank/results/cann_lowrank_summary.md``: the
#: smallest rank that keeps the bump-position error below ~5 mrad
#: (about 0.3° on a ring of circumference 2π).
ACCL_DEFAULT_K: dict[tuple[str, str], int] = {
    ("CANN1D", "fast"): 8,
    ("CANN1D", "ultra-fast"): 1,
    ("CANN2D", "fast"): 32,
    ("CANN2D", "ultra-fast"): 4,
}


# ---------------------------------------------------------------------------
# Spectrum-based rank picker (used by ``accl_mode="auto"``)
# ---------------------------------------------------------------------------


def _pick_k_for_err_target(
    spectrum: np.ndarray,
    target_err_mrad: float,
) -> int:
    """Pick the smallest rank whose predicted bump-tracking error is
    within ``target_err_mrad``.

    The estimate is calibrated against the ``canns_lowrank`` benchmark
    (see ``benchmarks/cann_lowrank/results/cann_lowrank_summary.md``):

    =========================== =====================
    ``target_err_mrad`` budget  heuristic used
    =========================== =====================
    ``>= 4.0``                   ``k = 1`` (always preserves bump tracking; the leading singular vector of a Gaussian kernel *is* the bump shape)
    ``>= 1.0``                   smallest k with ``captured_energy >= 0.5``
    ``>= 0.5``                   smallest k with ``captured_energy >= 0.9``
    ``>= 0.1``                   smallest k with ``captured_energy >= 0.99``
    ``< 0.1``                    ``-1`` (no lowrank can satisfy; fall back to dense)
    =========================== =====================

    Args:
        spectrum: Singular values of ``conn_mat`` (descending), as
            returned by ``np.linalg.svd(..., full_matrices=False)``.
        target_err_mrad: User-specified maximum allowed bump-position
            error in milliradians.

    Returns:
        The rank to use. Returns ``-1`` (the dense sentinel) if no
        low-rank truncation satisfies the budget.
    """
    spectrum = np.asarray(spectrum, dtype=np.float64)
    if spectrum.size == 0:
        return -1
    if target_err_mrad >= 4.0:
        return 1
    if target_err_mrad >= 1.0:
        threshold = 0.5
    elif target_err_mrad >= 0.5:
        threshold = 0.9
    elif target_err_mrad >= 0.1:
        threshold = 0.99
    else:
        return -1

    sq = spectrum**2
    total = float(sq.sum())
    if total <= 0:
        # All-zero spectrum: any k gives the exact (zero) approximation,
        # so fall through to "use smallest k = 1".
        return 1
    cum = np.cumsum(sq) / total
    # First k such that cum[k-1] >= threshold; default to dense.
    idx = int(np.searchsorted(cum, threshold, side="left")) + 1
    return min(idx, spectrum.size)


# ---------------------------------------------------------------------------
# Backend interface
# ---------------------------------------------------------------------------


@runtime_checkable
class IrecBackend(Protocol):
    """Callable strategy: ``backend(r) -> Irec``.

    Concrete backends must carry:

    - ``mode`` (str): the public ``accl_mode`` value
    - ``k``   (int): the effective rank (``-1`` for full-rank / exact)
    - ``U``, ``V``, ``K_fft``: the pre-computed factors (any of them
      may be ``None`` for backends that do not use them). The model
      exposes these via properties for backward compatibility.
    """

    mode: str
    k: int

    def __call__(self, r: jax.Array) -> jax.Array: ...


# ---------------------------------------------------------------------------
# Concrete backends
# ---------------------------------------------------------------------------


class DenseIrec:
    """Full-rank ``Irec = W @ r``. The default backend; no pre-computation.

    This is the reference implementation: it stores the full
    ``conn_mat`` and computes the matvec as ``W @ r`` every step.
    Cost is ``O(n²)`` in the number of neurons, which dominates the
    total step time for large ``num`` (1D) or ``length`` (2D). For
    small networks the constant-factor overhead of the
    low-rank / FFT backends exceeds their savings, so this backend
    is also the fastest option up to roughly ``num = 128`` (1D) or
    ``length = 8`` (2D).

    Use this mode when:

    - ``num <= 128`` (1D) or ``length <= 8`` (2D), where the speedup
      from any approximation is small.
    - You need the matvec to be **bit-exact** with the textbook
      formulation (e.g. for accuracy benchmarks comparing modes).
    - The model parameters are exotic (e.g. non-Gaussian
      ``conn_mat``) and the SVD/FFT structure does not apply.
    """

    mode = "normal"
    k = -1

    def __init__(self, conn_mat):
        self.W = conn_mat
        # Exposed for inspection parity with the low-rank / FFT backends.
        self.U = None
        self.V = None
        self.K_fft = None

    def __call__(self, r):
        return self.W @ r


class LowRankIrec:
    """``Irec = U @ (V.T @ r)`` from the truncated SVD of ``conn_mat``.

    The factors are *balanced*: ``U = U_full[:, :k] * sqrt(S[:k])`` and
    ``V = Vt[:k, :].T * sqrt(S[:k])``, so the product ``U @ V.T``
    recovers the rank-``k`` SVD truncation of ``conn_mat`` exactly
    (in the original SVD basis) up to the spectral tail past ``k``.

    Cost is ``O(n · k)`` per step instead of the dense ``O(n²)``,
    which is a 30–245× speedup at the recommended ranks for
    ``num >= 512``. The error is bounded by the discarded singular
    values (operator-norm error ``= S[k+1]``).

    Use this mode when:

    - You want the **recommended** production speedup. The default
      ranks (``CANN1D`` k=8, ``CANN2D`` k=32) keep the bump-position
      error below ~5 mrad in the ``canns_lowrank`` benchmark.
    - The kernel is approximately low-rank (smooth, decaying). The
      Gaussian distance kernel that canns's ``make_conn`` builds is
      one such case; its singular values decay fast.
    - You are willing to accept a small (sub-mrad to a few-mrad)
      error in the bump dynamics.

    For ``"auto"`` mode, the rank is selected from the SVD spectrum
    to satisfy the user's ``accl_target_err_mrad`` budget — see
    :func:`_pick_k_for_err_target`.
    """

    def __init__(self, U: jax.Array, V: jax.Array, k: int, mode: str):
        self.U = U
        self.V = V
        self.k = k
        self.mode = mode
        # Unused for this backend.
        self.K_fft = None

    def __call__(self, r):
        return self.U @ (self.V.T @ r)


class CirculantFFTIrec1D:
    """``Irec = real(ifft(K_fft * fft(r)))`` for a clean circulant ring.

    Exploits the fact that a right-circulant matrix's matvec is a
    circular convolution, which the convolution theorem reduces to
    a pointwise product in the Fourier domain. Cost is ``O(n log n)``
    per step, independent of the (now-irrelevant) ``k`` parameter.

    Requires ``conn_mat`` to be right-circulant, which is true **only
    on a uniform grid with ``endpoint=False``**. The canns default
    uses ``endpoint=True``, which duplicates the wrap point and
    breaks the circulant structure; in that case
    :func:`make_irec_backend` falls back to :class:`DenseIrec` with
    a warning. To use this backend, override the grid before
    requesting FFT mode:

    .. code-block:: python

        m = CANN1D(num=512, accl_mode="normal")
        m.x = bm.linspace(-bm.pi, bm.pi, 512, endpoint=False)
        m.conn_mat = m.make_conn()
        m.set_accl_mode("fft")

    Use this mode when:

    - The grid is uniform and ``endpoint=False`` (clean ring), and
    - You need the matvec to be **bit-exact** (not approximate like
      low-rank) at O(n log n) cost.

    The FFT form is **not** an approximation; on a clean circulant
    it equals ``conn_mat @ r`` to machine precision (fp32: rel err
    ``< 1e-5``).
    """

    mode = "fft"
    k = -1

    def __init__(self, K_fft: jax.Array):
        self.K_fft = K_fft
        # Unused for this backend; exposed as ``None`` for inspection.
        self.U = None
        self.V = None

    def __call__(self, r):
        return jnp.real(jnp.fft.ifft(self.K_fft * jnp.fft.fft(r)))


class CirculantFFTIrec2D:
    """``Irec = real(ifft2(K_fft2 * fft2(r_2d))).ravel()`` for a clean torus.

    2D analogue of :class:`CirculantFFTIrec1D`: a doubly-circulant
    matrix's matvec is a 2D circular convolution, computed exactly
    via 2D FFT in ``O(L² log L)`` per step. The 1D reshape to/from
    the 2D grid happens inside ``__call__``; the input/output are
    flat ``(length²,)`` to match the model's
    :attr:`_accel_Irec` contract.

    Same setup caveat as the 1D backend: requires a uniform
    ``endpoint=False`` grid. The canns default
    ``endpoint=True`` grid is not doubly-circulant, so
    :func:`make_irec_backend` falls back to :class:`DenseIrec`
    with a warning. The setup is identical to the 1D case — just
    assign ``m.x = bm.linspace(-bm.pi, bm.pi, length,
    endpoint=False)``, rebuild ``m.conn_mat``, then call
    ``m.set_accl_mode("fft")``.
    """

    mode = "fft"
    k = -1

    def __init__(self, K_fft2: jax.Array, length: int):
        self.K_fft2 = K_fft2
        self.length = length
        self.U = None
        self.V = None
        # Alias for inspection parity with the 1D backend.
        self.K_fft = K_fft2

    def __call__(self, r):
        r_2d = r.reshape(self.length, self.length)
        out_2d = jnp.real(jnp.fft.ifft2(self.K_fft2 * jnp.fft.fft2(r_2d)))
        return out_2d.ravel()


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
            default rank in :data:`ACCL_DEFAULT_K` for
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
    "ACCL_MODES",
    "ACCL_DEFAULT_K",
    "_pick_k_for_err_target",
    "IrecBackend",
    "DenseIrec",
    "LowRankIrec",
    "CirculantFFTIrec1D",
    "CirculantFFTIrec2D",
    "make_irec_backend",
]
