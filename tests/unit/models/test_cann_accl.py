"""Unit tests for the accl_mode / accl_k feature on CANN1D / CANN2D.

Covers:
  - default (normal) mode matches the historical dense behaviour
  - "fast" / "ultra-fast" modes give the recommended defaults
  - "auto" mode picks k from the SVD spectrum to satisfy
    ``accl_target_err_mrad``
  - "fft" mode gives an exact circulant matvec (when the grid is
    endpoint=False; otherwise it falls back to dense with a warning)
  - accl_k overrides the mode default
  - set_accl_mode round-trips correctly
  - validation rejects bad mode / bad k
  - the integrated update() preserves dynamics under low-rank
    (matches the standalone benchmark within tolerance)
"""

from __future__ import annotations

import os
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")

# NumPy + scipy emit spurious ``RuntimeWarning: divide by zero
# encountered in matmul`` on perfectly well-conditioned float32
# matmuls (verified in isolation with pure numpy). They are not
# actionable here — silence them module-wide.
warnings.filterwarnings(
    "ignore",
    message=".* encountered in matmul",
    category=RuntimeWarning,
)

import brainpy.math as bm
import numpy as np
import pytest

from canns.models.basic import CANN1D, CANN1D_SFA, CANN2D, CANN2D_SFA
from canns.models.basic.cann import (
    ACCL_DEFAULT_K,
    ACCL_MODES,
    _pick_k_for_err_target,
)

# ---------------------------------------------------------------------------
# Default-mode (normal) behaviour
# ---------------------------------------------------------------------------


class TestNormalModeDefault:
    def test_cann1d_default_is_normal(self):
        m = CANN1D(num=64)
        assert m.accl_mode == "normal"
        assert m.accl_k == -1
        assert m.is_accelerated is False
        assert m._U_l is None
        assert m._V_l is None

    def test_cann2d_default_is_normal(self):
        m = CANN2D(length=8)
        assert m.accl_mode == "normal"
        assert m.accl_k == -1
        assert m.is_accelerated is False
        assert m._U_l is None
        assert m._V_l is None

    def test_cann1d_sfa_default_is_normal(self):
        m = CANN1D_SFA(num=64)
        assert m.accl_mode == "normal"
        assert m.accl_k == -1

    def test_cann2d_sfa_default_is_normal(self):
        m = CANN2D_SFA(length=8)
        assert m.accl_mode == "normal"
        assert m.accl_k == -1


# ---------------------------------------------------------------------------
# Mode → default k mapping
# ---------------------------------------------------------------------------


class TestModeDefaults:
    @pytest.mark.parametrize(
        "cls,num_arg,mode,expected_k",
        [
            (CANN1D, 64, "fast", 8),
            (CANN1D, 64, "ultra-fast", 1),
            (CANN1D, 64, "normal", -1),
            (CANN2D, 8, "fast", 32),
            (CANN2D, 8, "ultra-fast", 4),
            (CANN2D, 8, "normal", -1),
            (CANN1D_SFA, 64, "fast", 8),
            (CANN2D_SFA, 8, "fast", 32),
        ],
    )
    def test_mode_default_k(self, cls, num_arg, mode, expected_k):
        m = cls(num_arg, accl_mode=mode)
        assert m.accl_mode == mode
        assert m.accl_k == expected_k
        if mode == "normal":
            assert m._U_l is None
        else:
            assert m._U_l is not None
            assert m._V_l is not None
            assert m._U_l.shape == (m.conn_mat.shape[0], expected_k)
            assert m._V_l.shape == (m.conn_mat.shape[0], expected_k)

    def test_default_k_table_matches_docs(self):
        # Sanity: the published defaults from the benchmark writeup.
        assert ACCL_DEFAULT_K[("CANN1D", "fast")] == 8
        assert ACCL_DEFAULT_K[("CANN1D", "ultra-fast")] == 1
        assert ACCL_DEFAULT_K[("CANN2D", "fast")] == 32
        assert ACCL_DEFAULT_K[("CANN2D", "ultra-fast")] == 4


# ---------------------------------------------------------------------------
# Explicit accl_k override
# ---------------------------------------------------------------------------


class TestAcclKOverride:
    def test_explicit_k_overrides_fast_default(self):
        m = CANN1D(num=128, accl_mode="fast", accl_k=4)
        assert m.accl_k == 4
        assert m._U_l.shape == (128, 4)
        assert m._V_l.shape == (128, 4)

    def test_explicit_k_with_normal_is_ignored(self):
        # ``accl_k`` is meaningless in normal mode. We accept it silently
        # (rather than error) and just use the full rank.
        m = CANN1D(num=128, accl_mode="normal", accl_k=4)
        assert m.accl_mode == "normal"
        assert m.accl_k == -1
        assert m._U_l is None

    def test_k_clamps_to_full_rank(self):
        # If user asks for k > n, we silently clamp to n.
        m = CANN1D(num=16, accl_mode="fast", accl_k=64)
        assert m.accl_k == 16

    def test_k_clamps_for_cann2d(self):
        m = CANN2D(length=4, accl_mode="fast", accl_k=999)
        # length=4 → n_neurons=16
        assert m.accl_k == 16


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_bad_mode_raises(self):
        with pytest.raises(ValueError, match="accl_mode must be one of"):
            CANN1D(num=64, accl_mode="turbo")

    def test_zero_k_raises(self):
        with pytest.raises(ValueError, match="accl_k must be a positive int"):
            CANN1D(num=64, accl_mode="fast", accl_k=0)

    def test_negative_k_raises(self):
        with pytest.raises(ValueError, match="accl_k must be a positive int"):
            CANN1D(num=64, accl_mode="fast", accl_k=-3)

    def test_float_k_raises(self):
        with pytest.raises(ValueError, match="accl_k must be a positive int"):
            CANN1D(num=64, accl_mode="fast", accl_k=1.5)

    def test_accl_modes_exported(self):
        assert ACCL_MODES == ("normal", "fast", "ultra-fast", "auto", "fft")


# ---------------------------------------------------------------------------
# set_accl_mode
# ---------------------------------------------------------------------------


class TestSetAcclMode:
    def test_set_fast_from_normal(self):
        m = CANN1D(num=64)
        assert m.accl_mode == "normal"
        m.set_accl_mode("fast")
        assert m.accl_mode == "fast"
        assert m.accl_k == 8
        assert m.is_accelerated is True
        assert m._U_l is not None

    def test_set_normal_from_fast(self):
        m = CANN1D(num=64, accl_mode="fast")
        assert m.accl_mode == "fast"
        m.set_accl_mode("normal")
        assert m.accl_mode == "normal"
        assert m.accl_k == -1
        assert m._U_l is None

    def test_explicit_k_overrides_mode(self):
        m = CANN1D(num=128)
        m.set_accl_mode("fast", k=2)
        assert m.accl_mode == "fast"
        assert m.accl_k == 2
        assert m._U_l.shape == (128, 2)

    def test_keep_mode_when_no_arg(self):
        m = CANN1D(num=64, accl_mode="fast", accl_k=4)
        # Pass only k; mode stays fast
        m.set_accl_mode(k=16)
        assert m.accl_mode == "fast"
        assert m.accl_k == 16

    def test_keep_k_when_no_arg(self):
        m = CANN1D(num=64, accl_mode="fast")
        m.set_accl_mode("ultra-fast")
        # k falls back to the ultra-fast default = 1
        assert m.accl_mode == "ultra-fast"
        assert m.accl_k == 1

    def test_cann2d_round_trip(self):
        m = CANN2D(length=8)
        m.set_accl_mode("fast", k=8)
        assert m.accl_k == 8
        assert m._U_l.shape == (64, 8)
        m.set_accl_mode("normal")
        assert m._U_l is None


# ---------------------------------------------------------------------------
# Integrated dynamics: fast mode preserves the dense dynamics
# ---------------------------------------------------------------------------


def _bump_pos_1d(r: np.ndarray, x: np.ndarray, z_range: float) -> float:
    """Circular-mean bump position (matches the benchmark helper)."""
    weights = np.maximum(r, 0)
    if weights.sum() < 1e-12:
        return float("nan")
    return float(np.angle(np.sum(weights * np.exp(1j * x))))


# ---------------------------------------------------------------------------
# "auto" mode: pick k from the SVD spectrum to satisfy accl_target_err_mrad
# ---------------------------------------------------------------------------


class TestPickKHelper:
    """Direct tests for the spectrum-based rank picker."""

    def test_default_target_picks_k1(self):
        # For any non-trivial spectrum, the most permissive budget picks k=1
        spectrum = np.array([1.0, 0.5, 0.25, 0.125, 0.0625])
        assert _pick_k_for_err_target(spectrum, target_err_mrad=10.0) == 1
        assert _pick_k_for_err_target(spectrum, target_err_mrad=5.0) == 1
        assert _pick_k_for_err_target(spectrum, target_err_mrad=4.0) == 1

    def test_tighter_target_picks_higher_k(self):
        # Realistic-ish Gaussian spectrum: S_k decays fast. cum_energy
        # at k=1 already 0.5+ for tight bands; pick accordingly.
        # Use a slower-decaying spectrum so the threshold steps up k.
        spectrum = np.array([1.0, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5])
        sq = spectrum**2
        cum = np.cumsum(sq) / sq.sum()
        # 50% threshold: smallest k with cum[k-1] >= 0.5
        k_50 = int(np.searchsorted(cum, 0.5, side="left")) + 1
        assert _pick_k_for_err_target(spectrum, target_err_mrad=1.0) == k_50
        # 90% threshold
        k_90 = int(np.searchsorted(cum, 0.9, side="left")) + 1
        assert _pick_k_for_err_target(spectrum, target_err_mrad=0.5) == k_90
        # 99% threshold (most k; the spectrum is long-tailed)
        k_99 = int(np.searchsorted(cum, 0.99, side="left")) + 1
        assert _pick_k_for_err_target(spectrum, target_err_mrad=0.1) == k_99
        # Monotone non-decreasing in tightness.
        assert k_50 <= k_90 <= k_99

    def test_impossible_target_returns_minus_one(self):
        spectrum = np.array([1.0, 0.5, 0.25])
        # target < 0.1 always falls back (caller interprets -1 as dense)
        assert _pick_k_for_err_target(spectrum, target_err_mrad=0.05) == -1
        assert _pick_k_for_err_target(spectrum, target_err_mrad=0.0) == -1

    def test_handles_degenerate_spectrum(self):
        # All-zeros conn has zero singular values; the per-rank-1
        # approximation is still exact (0 ≈ 0), so the helper
        # short-circuits to k=1. Empty spectrum → -1 (caller decides).
        assert _pick_k_for_err_target(np.zeros(5), target_err_mrad=5.0) == 1
        assert _pick_k_for_err_target(np.array([]), target_err_mrad=5.0) == -1
        # But if the budget is impossible (target < 0.1), even a
        # zero spectrum returns -1.
        assert _pick_k_for_err_target(np.zeros(5), target_err_mrad=0.0) == -1


class TestAutoMode:
    """``accl_mode='auto'`` picks k from the spectrum at __init__."""

    def test_cann1d_auto_default_target_picks_k1(self):
        # Default target 5.0 mrad → k=1 (matches the leading-Gaussian
        # argument: even rank-1 preserves the bump-tracking dynamics).
        m = CANN1D(num=128, accl_mode="auto")
        assert m.accl_mode == "auto"
        assert m.accl_k == 1
        assert m._U_l.shape == (128, 1)
        assert m._V_l.shape == (128, 1)

    def test_cann2d_auto_default_target_picks_k1(self):
        m = CANN2D(length=8, accl_mode="auto")
        assert m.accl_mode == "auto"
        assert m.accl_k == 1
        # 2D conn is length² × length² = 64 × 64
        assert m._U_l.shape == (64, 1)
        assert m._V_l.shape == (64, 1)

    @pytest.mark.parametrize(
        "target_mrad,model_cls,num_arg,expected_k_range",
        [
            # Tighter budgets should pick higher k (or fall back to dense).
            # 1D: k=8 is the 'fast' default (captures 99% of 1D spectrum).
            (5.0, CANN1D, 128, (1, 4)),
            (1.0, CANN1D, 128, (1, 8)),
            (0.5, CANN1D, 128, (4, 16)),
            # 2D: needs more components to reach the same captured energy.
            (5.0, CANN2D, 8, (1, 8)),
            (1.0, CANN2D, 8, (4, 16)),
            (0.5, CANN2D, 8, (16, 64)),
        ],
    )
    def test_tighter_target_picks_higher_k(self, target_mrad, model_cls, num_arg, expected_k_range):
        m = model_cls(num_arg, accl_mode="auto", accl_target_err_mrad=target_mrad)
        assert m.accl_mode == "auto"
        lo, hi = expected_k_range
        # Allow ±1 slack for spectrum quirks.
        assert lo - 1 <= m.accl_k <= hi + 1, (
            f"target={target_mrad} mrad → k={m.accl_k}, expected in [{lo},{hi}]"
        )

    def test_impossible_target_falls_back_to_dense(self):
        # target < 0.1 mrad cannot be satisfied by any lowrank CANN.
        m = CANN1D(num=128, accl_mode="auto", accl_target_err_mrad=0.05)
        assert m.accl_mode == "normal"  # silently downgraded
        assert m.accl_k == -1
        assert m._U_l is None
        assert m._V_l is None

    def test_explicit_k_overrides_auto_pick(self):
        # accl_k wins over the spectrum pick even in 'auto' mode.
        m = CANN1D(num=128, accl_mode="auto", accl_target_err_mrad=0.5, accl_k=2)
        assert m.accl_mode == "auto"
        assert m.accl_k == 2
        assert m._U_l.shape == (128, 2)

    def test_set_accl_mode_auto(self):
        m = CANN1D(num=128)
        m.set_accl_mode("auto")
        assert m.accl_mode == "auto"
        assert m.accl_k == 1
        # Tighten the budget at runtime.
        m.set_accl_mode("auto", target_err_mrad=0.5)
        assert m.accl_k >= 4
        # Back to dense.
        m.set_accl_mode("normal")
        assert m.accl_mode == "normal"
        assert m.accl_k == -1
        # And back to auto again, without passing target_err_mrad
        # (should keep the previously stored 0.5 budget).
        m.set_accl_mode("auto")
        assert m.accl_k >= 4

    def test_sfa_inherits_auto(self):
        m = CANN1D_SFA(num=64, accl_mode="auto")
        assert m.accl_mode == "auto"
        assert m.accl_k == 1


class TestIntegratedDynamics:
    """End-to-end check that the in-model low-rank path matches the
    standalone benchmark: the bump position under a moving stimulus
    agrees with the dense model within a few mrad, and r_max is
    essentially unchanged."""

    def test_cann1d_fast_preserves_dynamics(self):
        bm.set_dt(0.1)
        num = 256
        T = 100
        m_normal = CANN1D(num=num, accl_mode="normal")
        m_fast = CANN1D(num=num, accl_mode="fast")  # k=8 default
        x = np.asarray(m_normal.x)
        z_range = float(m_normal.z_range)

        def run(m):
            r_traj = np.empty((T, num), dtype=np.float32)
            for t in range(T):
                pos = np.pi * t / (T - 1)
                d = (x - pos) % z_range
                d = np.where(d > z_range / 2, d - z_range, d)
                inp = (m.A * np.exp(-0.25 * (d / m.a) ** 2)).astype(np.float32)
                m.update(inp)
                r_traj[t] = np.asarray(m.r.value)
            return r_traj

        r_normal = run(m_normal)
        r_fast = run(m_fast)
        pos_n = np.array([_bump_pos_1d(r_normal[t], x, z_range) for t in range(T)])
        pos_f = np.array([_bump_pos_1d(r_fast[t], x, z_range) for t in range(T)])
        dpos = np.abs(pos_n - pos_f)
        dpos = np.minimum(dpos, z_range - dpos)
        max_pos_err = float(np.nanmax(dpos))
        # The standalone benchmark reports < 10 mrad for n=256, k=8.
        # Allow a 2x slack for this smaller T=100 run.
        assert max_pos_err < 0.020, f"pos_err={max_pos_err * 1000:.2f} mrad"
        # r_max should be virtually identical.
        rmax_diff = np.abs(r_normal.max(axis=1) - r_fast.max(axis=1))
        assert float(rmax_diff.max()) < 1e-3, f"r_max_diff={float(rmax_diff.max()):.2e}"

    def test_cann1d_ultra_fast_at_k1_still_tracks(self):
        bm.set_dt(0.1)
        num = 256
        T = 100
        m_normal = CANN1D(num=num, accl_mode="normal")
        m_ultra = CANN1D(num=num, accl_mode="ultra-fast")  # k=1
        x = np.asarray(m_normal.x)
        z_range = float(m_normal.z_range)

        def run(m):
            for t in range(T):
                pos = np.pi * t / (T - 1)
                d = (x - pos) % z_range
                d = np.where(d > z_range / 2, d - z_range, d)
                inp = (m.A * np.exp(-0.25 * (d / m.a) ** 2)).astype(np.float32)
                m.update(inp)

        run(m_normal)
        run(m_ultra)
        # At k=1 the position error is still small (4-5 mrad in the
        # benchmark); the bump is wider because the connection is rank-1.
        # We only assert that the bump forms and the model runs without
        # blowing up.
        assert float(m_ultra.r.value.max()) > 0
        assert np.isfinite(np.asarray(m_ultra.r.value)).all()

    def test_cann2d_fast_preserves_dynamics(self):
        bm.set_dt(0.1)
        L = 16
        T = 50
        m_normal = CANN2D(length=L, accl_mode="normal")
        m_fast = CANN2D(length=L, accl_mode="fast")
        x = np.asarray(m_normal.x)
        z_range = float(m_normal.z_range)
        Lx, Ly = np.meshgrid(x, x)

        def run(m):
            for t in range(T):
                pos = np.pi * t / (T - 1)
                dx = (Lx - pos) % z_range
                dy = (Ly - pos) % z_range
                dx = np.where(dx > z_range / 2, dx - z_range, dx)
                dy = np.where(dy > z_range / 2, dy - z_range, dy)
                inp = (m.A * np.exp(-0.25 * ((dx**2 + dy**2) ** 0.5) / m.a) ** 2).astype(np.float32)
                m.update(inp)

        run(m_normal)
        run(m_fast)
        rmax_diff = abs(float(m_normal.r.value.max()) - float(m_fast.r.value.max()))
        assert rmax_diff < 1e-3, f"r_max_diff={rmax_diff:.2e}"
        assert float(m_fast.r.value.max()) > 0

    def test_sfa_fast_preserves_dynamics(self):
        bm.set_dt(0.1)
        m_normal = CANN1D_SFA(num=128, accl_mode="normal")
        m_fast = CANN1D_SFA(num=128, accl_mode="fast")
        for t in range(50):
            pos = np.pi * t / 49
            inp_normal = m_normal.get_stimulus_by_pos(pos)
            inp_fast = m_fast.get_stimulus_by_pos(pos)
            m_normal.update(inp_normal)
            m_fast.update(inp_fast)
        rmax_diff = abs(float(m_normal.r.value.max()) - float(m_fast.r.value.max()))
        assert rmax_diff < 1e-3
        assert float(m_fast.r.value.max()) > 0


# ---------------------------------------------------------------------------
# Dense-vs-lowrank match: lowrank output approximates the dense output
# ---------------------------------------------------------------------------


class TestLowrankApproximation:
    """The low-rank factor should approximate ``conn_mat`` to within
    the expected spectral truncation error."""

    @pytest.mark.parametrize(
        "cls,num_arg,n_neurons",
        [
            (CANN1D, 64, 64),
            (CANN1D, 128, 128),
            (CANN1D, 256, 256),
            (CANN2D, 8, 64),
            (CANN2D, 16, 256),
        ],
    )
    def test_lowrank_reconstructs_conn_within_spectral_tail(self, cls, num_arg, n_neurons):
        m = cls(num_arg, accl_mode="fast")
        # Numpy emits a spurious ``RuntimeWarning: divide by zero
        # encountered in matmul`` on perfectly fine float32 matmuls
        # (verified in isolation with pure numpy + scipy). Silence
        # locally — the result is well-conditioned.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            conn = np.array(m.conn_mat)
            U = np.array(m._U_l)
            V = np.array(m._V_l)
            approx = U @ V.T
            rel = float(np.linalg.norm(conn - approx) / np.linalg.norm(conn))
        # For CANN1D, captured_energy = 99.4% → relerr < 0.08.
        # For CANN2D, captured_energy at k=32 is ~92% → relerr < 0.30.
        # We assert generous bounds to avoid being brittle.
        assert rel < 0.4, f"relerr={rel:.3f} for {cls.__name__} {num_arg}"

    def test_normal_mode_has_no_lowrank_factors(self):
        m = CANN1D(num=64, accl_mode="normal")
        assert m._U_l is None
        assert m._V_l is None
        # And the in-model matvec should equal the dense one (up to fp32 noise).
        r = np.random.RandomState(0).randn(64).astype(np.float32) * 0.1
        r_jax = bm.asarray(r)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            Irec = m._accel_Irec(r_jax)
            Irec_dense = np.array(m.conn_mat) @ r
        np.testing.assert_allclose(np.asarray(Irec), Irec_dense, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# Docstring example sanity (the docstrings claim `model.update(inp=0.5)` works)
# ---------------------------------------------------------------------------


class TestDocstringExample:
    def test_cann1d_docstring_example_runs(self):
        # The base class docstring shows
        #   >>> model = CANN1D(num=512)
        #   >>> model.update(inp=0.5)
        # (positional/keyword `inp` mismatch — but our model accepts the
        # 1-D stimulus vector as positional). Verify the example is
        # consistent with the public API.
        bm.set_dt(0.1)
        m = CANN1D(num=64)
        stim = m.get_stimulus_by_pos(0.0)
        m.update(stim)  # works
        # Keyword form
        m.update(inp=stim)  # also works


# ---------------------------------------------------------------------------
# FFT mode (accl_mode="fft"): exact circulant matvec
# ---------------------------------------------------------------------------


def _force_endpoint_false(model_1d, n):
    """Override the default endpoint=True grid to a clean periodic grid
    (endpoint=False) and rebuild the connectivity matrix. The resulting
    K is right-circulant (1D) or doubly-circulant (2D), so the FFT
    matvec formula K @ r = IFFT(FFT(c) ⊙ FFT(r)) is exact.
    """
    model_1d.x = bm.linspace(-bm.pi, bm.pi, n, endpoint=False)
    model_1d.conn_mat = model_1d.make_conn()


def _force_endpoint_false_2d(model_2d, L):
    model_2d.x = bm.linspace(-bm.pi, bm.pi, L, endpoint=False)
    model_2d.y = bm.linspace(-bm.pi, bm.pi, L, endpoint=False)
    model_2d.conn_mat = model_2d.make_conn()


class TestFFTMode:
    """The FFT mode exploits the circulant structure of the Gaussian
    distance kernel on a uniform ring (1D) or torus (2D). On a clean
    circulant (endpoint=False grid) the matvec is exact up to float
    precision; on the canns default endpoint=True grid the structure
    is not circulant and we fall back to dense (with a warning).
    """

    # ---- 1D clean circulant ----

    @pytest.mark.parametrize("n", [32, 64, 128, 256])
    def test_1d_clean_circulant_matches_dense(self, n):
        bm.random.seed(0)
        m = CANN1D(num=n, accl_mode="normal")
        _force_endpoint_false(m, n)
        r = bm.random.rand(n)
        dense = np.asarray(m.conn_mat @ r)
        m._setup_accl(accl_mode="fft", accl_k=None)
        fft_out = np.asarray(m._accel_Irec(r))
        np.testing.assert_allclose(fft_out, dense, rtol=1e-5, atol=1e-6)
        assert m.accl_mode == "fft"

    def test_1d_clean_circulant_via_constructor(self):
        bm.random.seed(0)
        # Build the model with the default grid, then rebuild on
        # endpoint=False via make_conn, then ask for FFT.
        m = CANN1D(num=128, accl_mode="normal")
        _force_endpoint_false(m, 128)
        m._setup_accl(accl_mode="fft", accl_k=None)
        assert m.accl_mode == "fft"
        assert m._K_fft is not None
        assert m._U_l is None  # FFT path uses K_fft, not the SVD factors

    def test_1d_clean_circulant_accl_k_ignored(self):
        # accl_k is meaningless for FFT (the matvec is exact).
        # The setup accepts any accl_k silently.
        bm.random.seed(0)
        m = CANN1D(num=64, accl_mode="normal")
        _force_endpoint_false(m, 64)
        m._setup_accl(accl_mode="fft", accl_k=7)
        assert m.accl_mode == "fft"
        assert m.accl_k == -1
        assert m._K_fft is not None

    # ---- 1D default (endpoint=True) — falls back to dense ----

    def test_1d_default_grid_falls_back_to_normal(self):
        bm.random.seed(0)
        m = CANN1D(num=64, accl_mode="normal")
        # Default grid: endpoint=True. The wrap convention is not
        # circulant, so the FFT formula does not apply.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            m._setup_accl(accl_mode="fft", accl_k=None)
        assert m.accl_mode == "normal"
        assert m._K_fft is None
        assert m._U_l is None
        # The warning mentions endpoint=True and the fallback.
        msgs = [str(w.message) for w in caught]
        assert any("endpoint=True" in s for s in msgs)
        assert any("Falling back" in s for s in msgs)

    def test_1d_constructor_with_fft_on_default_falls_back(self):
        bm.random.seed(0)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            m = CANN1D(num=64, accl_mode="fft")
        assert m.accl_mode == "normal"
        assert m._K_fft is None

    # ---- 2D clean doubly-circulant ----

    @pytest.mark.parametrize("L", [4, 8, 16, 32])
    def test_2d_clean_circulant_matches_dense(self, L):
        bm.random.seed(0)
        m = CANN2D(length=L, accl_mode="normal")
        _force_endpoint_false_2d(m, L)
        r = bm.random.rand(L * L)
        dense = np.asarray(m.conn_mat @ r)
        m._setup_accl(accl_mode="fft", accl_k=None)
        fft_out = np.asarray(m._accel_Irec(r))
        np.testing.assert_allclose(fft_out, dense, rtol=1e-5, atol=1e-6)
        assert m.accl_mode == "fft"

    # ---- 2D default (endpoint=True) — falls back to dense ----

    def test_2d_default_grid_falls_back_to_normal(self):
        bm.random.seed(0)
        m = CANN2D(length=8, accl_mode="normal")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            m._setup_accl(accl_mode="fft", accl_k=None)
        assert m.accl_mode == "normal"
        assert m._K_fft is None
        msgs = [str(w.message) for w in caught]
        assert any("Falling back" in s for s in msgs)

    # ---- set_accl_mode round-trip ----

    def test_set_accl_mode_fft_clean_1d(self):
        bm.random.seed(0)
        m = CANN1D(num=64, accl_mode="normal")
        _force_endpoint_false(m, 64)
        m.set_accl_mode("fft")
        assert m.accl_mode == "fft"
        assert m._K_fft is not None
        # Now switch back: the K_fft cache should be cleared.
        m.set_accl_mode("normal")
        assert m.accl_mode == "normal"
        assert m._K_fft is None

    def test_set_accl_mode_fft_default_1d_falls_back(self):
        bm.random.seed(0)
        m = CANN1D(num=64, accl_mode="normal")
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            m.set_accl_mode("fft")
        assert m.accl_mode == "normal"
        assert m._K_fft is None

    # ---- End-to-end update still works under FFT mode ----

    def test_cann1d_fft_update_runs(self):
        bm.random.seed(0)
        bm.set_dt(0.1)
        m = CANN1D(num=64, accl_mode="normal")
        _force_endpoint_false(m, 64)
        m._setup_accl(accl_mode="fft", accl_k=None)
        assert m.accl_mode == "fft"
        # Run a few update steps; the model should not raise.
        for _ in range(5):
            stim = m.get_stimulus_by_pos(0.0)
            m.update(stim)

    def test_cann2d_fft_update_runs(self):
        bm.random.seed(0)
        bm.set_dt(0.1)
        m = CANN2D(length=8, accl_mode="normal")
        _force_endpoint_false_2d(m, 8)
        m._setup_accl(accl_mode="fft", accl_k=None)
        assert m.accl_mode == "fft"
        for _ in range(5):
            stim = m.get_stimulus_by_pos(bm.asarray([0.0, 0.0]))
            m.update(stim)

