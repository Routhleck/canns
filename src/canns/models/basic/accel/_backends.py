"""Concrete :class:`IrecBackend` strategies for CANN matvec acceleration.

Each backend is a self-contained callable that carries its own
pre-computed factors and exposes a single ``__call__(r) -> Irec``
entry point. ``canns.models.basic.cann.BaseCANN1D/2D`` hold one of
these on ``self.irec_backend`` and just call it from ``update()``.

Adding a new acceleration mode (e.g. block-circulant, Chebyshev,
random projection) means:

1. Adding a new class here (mirror the existing shape).
2. Adding a new value to :data:`ACCL_MODES` in :mod:`.accel`.
3. Adding a dispatch branch in :func:`make_irec_backend` in
   :mod:`.factory`.

The model class does not change.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Dense baseline
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


# ---------------------------------------------------------------------------
# Low-rank (SVD)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Circulant FFT
# ---------------------------------------------------------------------------


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


__all__ = [
    "DenseIrec",
    "LowRankIrec",
    "CirculantFFTIrec1D",
    "CirculantFFTIrec2D",
]
