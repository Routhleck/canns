"""Strategy-pattern backends for the CANN recurrent matvec ``Irec = W @ r``.

This subpackage holds everything related to accelerating the
recurrent matvec in ``canns.models.basic.cann``. The model class
holds one backend on ``self.irec_backend`` and just calls it from
``update()``; the choice of strategy lives entirely here.

Layout
------

==================  ============================================
Module               Contents
==================  ============================================
:mod:`._protocol`    :class:`IrecBackend` Protocol (duck type)
:mod:`._backends`    :class:`DenseIrec`, :class:`LowRankIrec`,
                     :class:`CirculantFFTIrec1D`, :class:`CirculantFFTIrec2D`
:mod:`._spectrum`    :func:`_pick_k_for_err_target` (used by ``"auto"``)
:mod:`._factory`     :func:`make_irec_backend` (single dispatch point)
==================  ============================================

The public surface re-exported here is the only thing the rest of
canns (and external users) should import. Internal files are
prefixed with ``_`` to mark them as implementation details.

For backward compatibility the public symbols
:data:`ACCL_MODES`, :data:`ACCL_DEFAULT_K`, and
:func:`_pick_k_for_err_target` are also re-exported from
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
``"fft"``           ``real(ifft(K_fft ⊙ fft(r)))``             **Yes**      25–50×        **Only** on a clean ring/torus with ``endpoint=False``. The canns default ``endpoint=True`` grid is not circulant, so this mode falls back to ``"normal"`` with a warning. See the table below for setup.
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

Three steps, all in this subpackage:

1. **Add a class** in :mod:`._backends` that holds whatever
   pre-computed factors you need and implements ``__call__(self,
   r)``. Mirror the existing :class:`DenseIrec` / :class:`LowRankIrec`
   / :class:`CirculantFFTIrec*` shape.
2. **Register the mode** here: add a new value to
   :data:`ACCL_MODES` (and to :data:`ACCL_DEFAULT_K` if your mode
   has per-family default ranks).
3. **Add a dispatch branch** in :func:`make_irec_backend` in
   :mod:`._factory`.

``cann.py`` does not need to change. The model only sees a callable
that takes ``r`` and returns ``Irec``.
"""

from __future__ import annotations

# Public constants — the canonical list of modes and the default
# rank for each (model family, mode) combination.
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

# Re-exports
from ._backends import (
    CirculantFFTIrec1D,
    CirculantFFTIrec2D,
    DenseIrec,
    LowRankIrec,
)
from ._factory import make_irec_backend
from ._protocol import IrecBackend
from ._spectrum import _pick_k_for_err_target

__all__ = [
    # Public constants
    "ACCL_MODES",
    "ACCL_DEFAULT_K",
    # Public backend classes (advanced users may want to inspect /
    # subclass these)
    "DenseIrec",
    "LowRankIrec",
    "CirculantFFTIrec1D",
    "CirculantFFTIrec2D",
    "IrecBackend",
    # Public factory
    "make_irec_backend",
    # Re-exported helper (also available from canns.models.basic.cann
    # for backward compatibility)
    "_pick_k_for_err_target",
]
