"""Spectrum-based rank picker for ``accl_mode="auto"``.

Given the singular values of ``conn_mat`` and a milliradian error
budget, :func:`_pick_k_for_err_target` returns the smallest rank
whose predicted bump-tracking error fits. The calibration is in
:mod:`canns.models.basic.benchmarks.cann_lowrank`.

This module is pure NumPy — no JAX / brainpy dependency — so the
picker can be unit-tested without a heavy runtime stack.
"""

from __future__ import annotations

import numpy as np


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


__all__ = ["_pick_k_for_err_target"]
