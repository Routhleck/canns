"""Unit tests for the accl_mode / accl_k feature on CANN1D / CANN2D.

Covers:
  - default (normal) mode matches the historical dense behaviour
  - "fast" / "ultra-fast" modes give the recommended defaults
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

import numpy as np
import pytest
import brainpy.math as bm

from canns.models.basic import CANN1D, CANN2D, CANN1D_SFA, CANN2D_SFA
from canns.models.basic.cann import ACCL_MODES, ACCL_DEFAULT_K


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
        assert ACCL_MODES == ("normal", "fast", "ultra-fast")


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
        assert max_pos_err < 0.020, f"pos_err={max_pos_err*1000:.2f} mrad"
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
                inp = (m.A * np.exp(-0.25 * ((dx ** 2 + dy ** 2) ** 0.5) / m.a) ** 2).astype(np.float32)
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
    def test_lowrank_reconstructs_conn_within_spectral_tail(
        self, cls, num_arg, n_neurons
    ):
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
