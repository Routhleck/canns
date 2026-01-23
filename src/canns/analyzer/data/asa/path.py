from __future__ import annotations

import numpy as np


def load_npz_any(path: str) -> dict:
    """Load .npz into a plain dict (allow_pickle=True)."""
    obj = np.load(path, allow_pickle=True)
    try:
        if hasattr(obj, "files"):
            return {k: obj[k] for k in obj.files}
        return dict(obj)
    finally:
        try:
            obj.close()
        except Exception:
            pass


def unwrap_container(x):
    """Unwrap 0-d object arrays that store a python object."""
    if isinstance(x, np.ndarray) and x.dtype == object and x.shape == ():
        return x.item()
    return x


def as_1d(x) -> np.ndarray:
    """Convert input to a 1D numpy array (robust to 0-d object containers)."""
    x = np.asarray(x)
    if x.dtype == object and x.shape == ():
        x = x.item()
        x = np.asarray(x)
    return np.asarray(x).ravel()


def find_times_box(dec: dict) -> tuple[np.ndarray | None, str | None]:
    """Try to find a 'times_box' / keep-index vector in decoding dict."""
    keys = [
        "times_box",
        "timesbox",
        "t_box",
        "idx_box",
        "index_box",
        "indices_box",
        "keep",
        "keep_idx",
        "keep_indices",
        "idx",
        "indices",
        "times",
        "t",
    ]
    for k in keys:
        if k in dec:
            tb = unwrap_container(dec[k])
            tb = np.asarray(tb)
            if tb.ndim == 1 or (tb.ndim == 2 and 1 in tb.shape):
                return as_1d(tb), k
    return None, None


def _is_2col(arr) -> bool:
    return isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[1] == 2


def find_coords_matrix(
    dec: dict,
    coords_key: str | None = None,
    prefer_box_fallback: bool = False,
) -> tuple[np.ndarray, str]:
    """Find a decoded coords matrix (N,D>=2) in decoding dict.

    IMPORTANT: to match your original test1 behavior, we ALWAYS prefer a true (N,2)
    angles matrix (e.g. key 'coords') if it exists, even if you set --use-box.

    Only when no (N,2) matrix exists do we fall back to >=2-col matrices (coordsbox, etc.).
    """
    dec = {k: unwrap_container(v) for k, v in dec.items()}

    if coords_key is not None:
        if coords_key not in dec:
            raise KeyError(f"--coords-key '{coords_key}' not found. keys={list(dec.keys())}")
        arr = np.asarray(dec[coords_key])
        if arr.ndim != 2 or arr.shape[1] < 2:
            raise ValueError(
                f"--coords-key '{coords_key}' must be 2D with >=2 cols, got {arr.shape}"
            )
        return arr, coords_key

    base_keys = ["coords", "theta", "thetas", "circular_coords", "cc", "decoded_coords"]
    box_keys = ["coordsbox", "coords_box", "coordsBox", "coords_box_full"]

    # Pass 1: prefer true (N,2) in base keys (this matches original test1)
    for k in base_keys:
        if k in dec and _is_2col(np.asarray(dec[k])):
            return np.asarray(dec[k]), k

    # Pass 2: accept true (N,2) in box keys
    for k in box_keys:
        if k in dec and _is_2col(np.asarray(dec[k])):
            return np.asarray(dec[k]), k

    # Pass 3: fall back to any matrix with >=2 cols
    search = (box_keys + base_keys) if prefer_box_fallback else (base_keys + box_keys)
    for k in search:
        if k in dec:
            arr = np.asarray(dec[k])
            if arr.ndim == 2 and arr.shape[1] >= 2:
                return arr, k

    # fallback: two 1D arrays
    cands1 = ["theta1", "th1", "phi1", "u", "circ1"]
    cands2 = ["theta2", "th2", "phi2", "v", "circ2"]
    for k1 in cands1:
        for k2 in cands2:
            if k1 in dec and k2 in dec:
                a = as_1d(dec[k1])
                b = as_1d(dec[k2])
                if len(a) == len(b) and len(a) > 0:
                    return np.stack([a, b], axis=1), f"{k1}+{k2}"

    raise KeyError(f"Cannot find decoded coords matrix. keys={list(dec.keys())}")


def resolve_time_slice(t: np.ndarray, tmin, tmax, imin, imax) -> tuple[int, int]:
    """Return [i0,i1) slice bounds using either index bounds or time bounds."""
    T = len(t)
    if imin is not None or imax is not None:
        i0 = 0 if imin is None else max(0, int(imin))
        i1 = T if imax is None else min(T, int(imax))
        return i0, i1

    if tmin is None and tmax is None:
        return 0, T

    tmin = t[0] if tmin is None else float(tmin)
    tmax = t[-1] if tmax is None else float(tmax)
    if tmax < tmin:
        tmin, tmax = tmax, tmin

    i0 = int(np.searchsorted(t, tmin, side="left"))
    i1 = int(np.searchsorted(t, tmax, side="right"))
    i0 = max(0, min(T, i0))
    i1 = max(0, min(T, i1))
    return i0, i1


def skew_transform(theta_2d: np.ndarray) -> np.ndarray:
    """Map (theta1,theta2) to skew coordinates used in the base parallelogram."""
    th1 = theta_2d[:, 0]
    th2 = theta_2d[:, 1]
    X = th1 + 0.5 * th2
    Y = (np.sqrt(3) / 2.0) * th2
    return np.stack([X, Y], axis=1)


def draw_base_parallelogram(ax):
    e1 = np.array([2 * np.pi, 0.0])
    e2 = np.array([np.pi, np.sqrt(3) * np.pi])
    P00 = np.array([0.0, 0.0])
    P10 = e1
    P01 = e2
    P11 = e1 + e2
    poly = np.vstack([P00, P10, P11, P01, P00])
    ax.plot(poly[:, 0], poly[:, 1], lw=1.2, color="0.35")


def parse_times_box_to_indices(times_box: np.ndarray, t_full: np.ndarray) -> tuple[np.ndarray, str]:
    """Convert times_box to integer indices into t_full."""
    tb = as_1d(times_box)
    T_full = len(t_full)

    if np.issubdtype(tb.dtype, np.integer):
        idx = tb.astype(int)
        kind = "index"
    else:
        # If values are basically integers and within range -> treat as indices
        if (
            np.all(np.isfinite(tb))
            and np.all(np.abs(tb - np.round(tb)) < 1e-6)
            and np.nanmax(tb) <= T_full + 1
        ):
            idx = np.round(tb).astype(int)
            kind = "index(float)"
        else:
            # Treat as timestamps -> map via searchsorted
            idx = np.searchsorted(t_full, tb, side="left").astype(int)
            idx = np.clip(idx, 0, T_full - 1)
            kind = "time"

    return idx, kind


def interp_coords_to_full(idx_map: np.ndarray, coords2: np.ndarray, T_full: int) -> np.ndarray:
    """Interpolate (K,2) circular coords back to full length (T_full,2)."""
    idx_map = np.asarray(idx_map).astype(int).ravel()
    coords2 = np.asarray(coords2, float)

    # sort & unique
    order = np.argsort(idx_map)
    idx_map = idx_map[order]
    coords2 = coords2[order]

    uniq_idx, uniq_pos = np.unique(idx_map, return_index=True)
    coords2 = coords2[uniq_pos]
    idx_map = uniq_idx

    # unwrap for interpolation stability
    ang = np.unwrap(coords2, axis=0)
    full_i = np.arange(T_full, dtype=float)

    out = np.zeros((T_full, 2), float)
    for d in range(2):
        out[:, d] = np.interp(full_i, idx_map.astype(float), ang[:, d])

    return np.mod(out, 2 * np.pi)


def align_coords_to_position(
    t_full: np.ndarray,
    x_full: np.ndarray,
    y_full: np.ndarray,
    coords2: np.ndarray,
    use_box: bool,
    times_box: np.ndarray | None,
    interp_to_full: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Align decoded coordinates to the original (x, y, t) trajectory.

    Parameters
    ----------
    t_full, x_full, y_full : np.ndarray
        Full-length trajectory arrays of shape (T,).
    coords2 : np.ndarray
        Decoded coordinates of shape (K, 2) or (T, 2).
    use_box : bool
        Whether to use ``times_box`` to align to the original trajectory.
    times_box : np.ndarray | None
        Time indices or timestamps corresponding to ``coords2`` when ``use_box=True``.
    interp_to_full : bool
        If True, interpolate decoded coords back to full length; otherwise return a subset.

    Returns
    -------
    tuple
        ``(t_aligned, x_aligned, y_aligned, coords_aligned, tag)`` where ``tag`` describes
        the alignment path used.

    Examples
    --------
    >>> t, x, y, coords2, tag = align_coords_to_position(  # doctest: +SKIP
    ...     t_full, x_full, y_full, coords2,
    ...     use_box=True, times_box=decoding["times_box"], interp_to_full=True
    ... )
    >>> coords2.shape[1]
    2
    """
    t_full = np.asarray(t_full).ravel()
    x_full = np.asarray(x_full).ravel()
    y_full = np.asarray(y_full).ravel()
    coords2 = np.asarray(coords2, float)

    T_full = len(t_full)

    if not use_box:
        if len(coords2) != T_full:
            raise ValueError(
                f"coords length {len(coords2)} != t length {T_full} "
                f"(set --use-box if you have times_box)"
            )
        return t_full, x_full, y_full, coords2, "full(no-box)"

    if times_box is None:
        if len(coords2) == T_full:
            return (
                t_full,
                x_full,
                y_full,
                coords2,
                "full(use-box but no times_box; treated as full)",
            )
        raise KeyError("use_box=True but times_box not found, and coords is not full-length.")

    idx_map, kind = parse_times_box_to_indices(times_box, t_full)

    # If coords already full and times_box also full, keep full
    if len(coords2) == T_full and len(idx_map) == T_full:
        return (
            t_full,
            x_full,
            y_full,
            coords2,
            f"full(coords already full; times_box kind={kind} ignored)",
        )

    if len(idx_map) != len(coords2):
        raise ValueError(f"times_box length {len(idx_map)} != coords length {len(coords2)}")

    order = np.argsort(idx_map)
    idx_map = idx_map[order]
    coords2 = coords2[order]

    if interp_to_full:
        coords_full = interp_coords_to_full(idx_map, coords2, T_full)
        return (
            t_full,
            x_full,
            y_full,
            coords_full,
            f"interp_to_full(times_box kind={kind}, K={len(idx_map)})",
        )

    idx_map = np.clip(idx_map, 0, T_full - 1)
    return (
        t_full[idx_map],
        x_full[idx_map],
        y_full[idx_map],
        coords2,
        f"subset(times_box kind={kind}, K={len(idx_map)})",
    )


def snake_wrap_trail_in_parallelogram(
    xy_base: np.ndarray, e1: np.ndarray, e2: np.ndarray
) -> np.ndarray:
    """Insert NaNs when the trail wraps across the torus fundamental domain."""
    xy_base = np.asarray(xy_base, float)
    if xy_base.ndim != 2 or xy_base.shape[1] != 2:
        raise ValueError(f"xy_base must be (T,2), got {xy_base.shape}")

    shifts = []
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            shifts.append(i * e1 + j * e2)
    shifts = np.asarray(shifts)  # (9,2)

    out = [xy_base[0]]
    for k in range(1, len(xy_base)):
        prev = xy_base[k - 1]
        cur = xy_base[k]

        disp = (cur[None, :] + shifts) - prev[None, :]
        d2 = np.sum(disp * disp, axis=1)
        best = shifts[np.argmin(d2)]

        if best[0] != 0.0 or best[1] != 0.0:
            out.append(np.array([np.nan, np.nan]))
        out.append(cur)

    return np.vstack(out)


def apply_angle_scale(coords2: np.ndarray, scale: str) -> np.ndarray:
    """Convert angle units to radians before wrapping.

    Parameters
    ----------
    coords2 : np.ndarray
        Angle array of shape (T, 2) in the given ``scale``.
    scale : {"rad", "deg", "unit", "auto"}
        ``rad``  : already in radians.
        ``deg``  : degrees -> radians.
        ``unit`` : unit circle in [0, 1] -> radians.
        ``auto`` : infer unit circle if values look like [0, 1].

    Returns
    -------
    np.ndarray
        Angles in radians.

    Examples
    --------
    >>> apply_angle_scale([[0.25, 0.5]], "unit")  # doctest: +SKIP
    """
    coords2 = np.asarray(coords2, float)
    if scale == "rad":
        return coords2
    if scale == "unit":
        return coords2 * (2 * np.pi)
    if scale == "deg":
        return np.deg2rad(coords2)
    if scale == "auto":
        # Heuristic: if values look like [0,1] or [-0.2,1.2], treat as unit circle coords.
        mn = float(np.nanmin(coords2))
        mx = float(np.nanmax(coords2))
        if mx <= 1.2 and mn >= -0.2:
            return coords2 * (2 * np.pi)
        return coords2
    raise ValueError(f"Unknown --scale option: {scale}")
