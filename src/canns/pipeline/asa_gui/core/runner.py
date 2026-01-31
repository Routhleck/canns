"""Pipeline execution wrapper for ASA GUI.

Provides synchronous pipeline execution that wraps canns.analyzer.data.asa APIs
and mirrors the TUI runner behavior for caching and artifacts.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .state import WorkflowState, resolve_path


@dataclass
class PipelineResult:
    """Result from pipeline execution."""

    success: bool
    artifacts: dict[str, Path]
    summary: str
    error: str | None = None
    elapsed_time: float = 0.0


class ProcessingError(RuntimeError):
    """Raised when a pipeline stage fails."""

    pass


class PipelineRunner:
    """Synchronous pipeline execution wrapper."""

    def __init__(self) -> None:
        self._asa_data: dict[str, Any] | None = None
        self._embed_data: np.ndarray | None = None
        self._aligned_pos: dict[str, np.ndarray] | None = None
        self._input_hash: str | None = None
        self._embed_hash: str | None = None
        self._mpl_ready: bool = False

    def has_preprocessed_data(self) -> bool:
        return self._embed_data is not None

    @property
    def embed_data(self) -> np.ndarray | None:
        return self._embed_data

    @property
    def aligned_pos(self) -> dict[str, np.ndarray] | None:
        return self._aligned_pos

    def reset_input(self) -> None:
        self._asa_data = None
        self._embed_data = None
        self._aligned_pos = None
        self._input_hash = None
        self._embed_hash = None

    def _json_safe(self, obj: Any) -> Any:
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, tuple):
            return [self._json_safe(v) for v in obj]
        if isinstance(obj, list):
            return [self._json_safe(v) for v in obj]
        if isinstance(obj, dict):
            return {str(k): self._json_safe(v) for k, v in obj.items()}
        if hasattr(obj, "item") and callable(obj.item):
            try:
                return obj.item()
            except Exception:
                return str(obj)
        return obj

    def _hash_bytes(self, data: bytes) -> str:
        return hashlib.md5(data).hexdigest()

    def _hash_file(self, path: Path) -> str:
        md5 = hashlib.md5()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                md5.update(chunk)
        return md5.hexdigest()

    def _hash_obj(self, obj: Any) -> str:
        payload = json.dumps(self._json_safe(obj), sort_keys=True, ensure_ascii=True).encode(
            "utf-8"
        )
        return self._hash_bytes(payload)

    def _ensure_matplotlib_backend(self) -> None:
        if self._mpl_ready:
            return
        try:
            import os

            os.environ.setdefault("MPLBACKEND", "Agg")
            import matplotlib

            try:
                matplotlib.use("Agg", force=True)
            except TypeError:
                matplotlib.use("Agg")
        except Exception:
            pass
        self._mpl_ready = True

    def _cache_dir(self, state: WorkflowState) -> Path:
        return self._results_dir(state) / ".asa_cache"

    def _results_dir(self, state: WorkflowState) -> Path:
        base = state.workdir / "Results"
        dataset_id = self._dataset_id(state)
        return base / dataset_id

    def results_dir(self, state: WorkflowState) -> Path:
        return self._results_dir(state)

    def _dataset_id(self, state: WorkflowState) -> str:
        try:
            input_hash = self._input_hash or self._compute_input_hash(state)
        except Exception:
            input_hash = "unknown"
        prefix = input_hash[:8]

        if state.input_mode == "asa":
            path = resolve_path(state, state.asa_file)
            stem = path.stem if path is not None else "asa"
            return f"{stem}_{prefix}"
        if state.input_mode == "neuron_traj":
            neuron_path = resolve_path(state, state.neuron_file)
            traj_path = resolve_path(state, state.traj_file)
            neuron_stem = neuron_path.stem if neuron_path is not None else "neuron"
            traj_stem = traj_path.stem if traj_path is not None else "traj"
            return f"{neuron_stem}_{traj_stem}_{prefix}"
        return f"{state.input_mode}_{prefix}"

    def _stage_cache_path(self, stage_dir: Path) -> Path:
        return stage_dir / "cache.json"

    def _load_cache_meta(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _write_cache_meta(self, path: Path, payload: dict[str, Any]) -> None:
        path.write_text(
            json.dumps(self._json_safe(payload), ensure_ascii=True, indent=2), encoding="utf-8"
        )

    def _stage_cache_hit(
        self, stage_dir: Path, expected_hash: str, required_files: list[Path]
    ) -> bool:
        if not all(p.exists() for p in required_files):
            return False
        meta = self._load_cache_meta(self._stage_cache_path(stage_dir))
        return meta.get("hash") == expected_hash

    def _compute_input_hash(self, state: WorkflowState) -> str:
        if state.input_mode == "asa":
            path = resolve_path(state, state.asa_file)
            if path is None:
                raise ProcessingError("ASA file not set.")
            return self._hash_obj({"mode": "asa", "file": self._hash_file(path)})
        if state.input_mode == "neuron_traj":
            neuron_path = resolve_path(state, state.neuron_file)
            traj_path = resolve_path(state, state.traj_file)
            if neuron_path is None or traj_path is None:
                raise ProcessingError("Neuron/trajectory files not set.")
            return self._hash_obj(
                {
                    "mode": "neuron_traj",
                    "neuron": self._hash_file(neuron_path),
                    "traj": self._hash_file(traj_path),
                }
            )
        return self._hash_obj({"mode": state.input_mode})

    def _load_npz_dict(self, path: Path) -> dict[str, Any]:
        data = np.load(path, allow_pickle=True)
        for key in ("persistence_result", "decode_result"):
            if key in data.files:
                return data[key].item()
        return {k: data[k] for k in data.files}

    def _build_spike_matrix_from_events(self, asa: dict[str, Any]) -> np.ndarray:
        if "t" not in asa:
            raise ProcessingError("asa dict missing key 't' for spike mode.")
        t = np.asarray(asa["t"])
        if t.ndim != 1:
            raise ProcessingError(f"asa['t'] must be 1D, got shape={t.shape}")
        total_steps = t.shape[0]

        raw = asa.get("spike")
        if raw is None:
            raise ProcessingError("asa dict missing key 'spike' for spike mode.")
        arr = np.asarray(raw)

        if isinstance(raw, np.ndarray) and arr.ndim == 2 and np.issubdtype(arr.dtype, np.number):
            if arr.shape[0] != total_steps:
                raise ProcessingError(
                    f"asa['spike'] matrix first dim {arr.shape[0]} != len(t)={total_steps}"
                )
            return arr.astype(float, copy=False)

        if isinstance(raw, np.ndarray) and arr.dtype == object and arr.size == 1:
            raw = arr.item()

        if isinstance(raw, dict):
            keys = sorted(raw.keys())
            spike_dict = {k: np.asarray(raw[k], dtype=float).ravel() for k in keys}
        elif isinstance(raw, (list, tuple)):
            keys = list(range(len(raw)))
            spike_dict = {i: np.asarray(raw[i], dtype=float).ravel() for i in keys}
        else:
            raise ProcessingError(
                "asa['spike'] must be a (T,N) numeric array, dict, or list-of-arrays for spike mode."
            )

        neuron_count = len(spike_dict)
        spike_mat = np.zeros((total_steps, neuron_count), dtype=float)
        if total_steps > 1:
            dt = float(t[1] - t[0])
        else:
            dt = 1.0
        t0 = float(t[0])

        for col, key in enumerate(keys):
            times = spike_dict[key]
            if times.size == 0:
                continue
            idx = np.rint((times - t0) / dt).astype(int)
            idx = idx[(idx >= 0) & (idx < total_steps)]
            if idx.size == 0:
                continue
            np.add.at(spike_mat[:, col], idx, 1.0)

        return spike_mat

    def _check_cancel(self, cancel_check: Callable[[], bool] | None) -> None:
        if cancel_check and cancel_check():
            raise ProcessingError("Cancelled by user")

    def run_preprocessing(
        self,
        state: WorkflowState,
        log_callback: Callable[[str], None],
        progress_callback: Callable[[int], None],
        cancel_check: Callable[[], bool] | None = None,
    ) -> PipelineResult:
        t0 = time.time()

        try:
            self._check_cancel(cancel_check)

            log_callback("Loading data...")
            progress_callback(10)
            asa_data = self._load_data(state)
            self._asa_data = asa_data
            self._aligned_pos = None
            self._input_hash = self._compute_input_hash(state)

            self._check_cancel(cancel_check)

            log_callback(f"Preprocessing with {state.preprocess_method}...")
            progress_callback(30)

            if state.preprocess_method == "embed_spike_trains":
                from canns.analyzer.data.asa import SpikeEmbeddingConfig, embed_spike_trains

                params = state.preprocess_params if state.preprocess_params else {}
                base_config = SpikeEmbeddingConfig()
                effective_params = {
                    "res": base_config.res,
                    "dt": base_config.dt,
                    "sigma": base_config.sigma,
                    "smooth": base_config.smooth,
                    "speed_filter": base_config.speed_filter,
                    "min_speed": base_config.min_speed,
                }
                effective_params.update(params)

                self._embed_hash = self._hash_obj(
                    {
                        "input_hash": self._input_hash,
                        "method": state.preprocess_method,
                        "params": effective_params,
                    }
                )
                cache_dir = self._cache_dir(state)
                cache_dir.mkdir(parents=True, exist_ok=True)
                cache_path = cache_dir / f"embed_{self._embed_hash}.npz"
                meta_path = cache_dir / f"embed_{self._embed_hash}.json"

                if cache_path.exists():
                    log_callback("♻️ Using cached embedding.")
                    cached = np.load(cache_path, allow_pickle=True)
                    self._embed_data = cached["embed_data"]
                    if {"x", "y", "t"}.issubset(set(cached.files)):
                        self._aligned_pos = {
                            "x": cached["x"],
                            "y": cached["y"],
                            "t": cached["t"],
                        }
                    progress_callback(100)
                    elapsed = time.time() - t0
                    return PipelineResult(
                        success=True,
                        artifacts={"embedding": cache_path},
                        summary=f"Preprocessing reused cached embedding in {elapsed:.1f}s",
                        elapsed_time=elapsed,
                    )

                config = SpikeEmbeddingConfig(**effective_params)

                log_callback("Running embed_spike_trains...")
                progress_callback(50)
                embed_result = embed_spike_trains(asa_data, config)

                if isinstance(embed_result, tuple):
                    embed_data = embed_result[0]
                    if len(embed_result) >= 4 and embed_result[1] is not None:
                        self._aligned_pos = {
                            "x": embed_result[1],
                            "y": embed_result[2],
                            "t": embed_result[3],
                        }
                else:
                    embed_data = embed_result

                self._embed_data = embed_data
                log_callback(f"Embed data shape: {embed_data.shape}")

                try:
                    payload = {"embed_data": embed_data}
                    if self._aligned_pos is not None:
                        payload.update(self._aligned_pos)
                    np.savez_compressed(cache_path, **payload)
                    self._write_cache_meta(
                        meta_path,
                        {
                            "hash": self._embed_hash,
                            "input_hash": self._input_hash,
                            "params": effective_params,
                        },
                    )
                except Exception as e:
                    log_callback(f"Warning: failed to cache embedding: {e}")
            else:
                log_callback("No preprocessing - using raw spike data")
                spike = asa_data.get("spike")
                self._embed_hash = self._hash_obj(
                    {
                        "input_hash": self._input_hash,
                        "method": state.preprocess_method,
                        "params": {},
                    }
                )

                if isinstance(spike, np.ndarray) and spike.ndim == 2:
                    self._embed_data = spike
                    log_callback(f"Using spike matrix shape: {spike.shape}")
                else:
                    log_callback(
                        "Warning: spike data is not a dense matrix, some analyses may fail"
                    )
                    self._embed_data = spike

            progress_callback(100)
            elapsed = time.time() - t0

            return PipelineResult(
                success=True,
                artifacts={},
                summary=f"Preprocessing completed in {elapsed:.1f}s",
                elapsed_time=elapsed,
            )

        except Exception as e:
            elapsed = time.time() - t0
            log_callback(f"Error: {e}")
            return PipelineResult(
                success=False,
                artifacts={},
                summary=f"Failed after {elapsed:.1f}s",
                error=str(e),
                elapsed_time=elapsed,
            )

    def run_analysis(
        self,
        state: WorkflowState,
        log_callback: Callable[[str], None],
        progress_callback: Callable[[int], None],
        cancel_check: Callable[[], bool] | None = None,
    ) -> PipelineResult:
        t0 = time.time()
        artifacts: dict[str, Path] = {}

        try:
            self._check_cancel(cancel_check)

            log_callback("Loading data...")
            progress_callback(10)
            asa_data = self._asa_data if self._asa_data is not None else self._load_data(state)
            if self._input_hash is None:
                self._input_hash = self._compute_input_hash(state)

            self._ensure_matplotlib_backend()

            log_callback(f"Running {state.analysis_mode} analysis...")
            progress_callback(40)

            mode = state.analysis_mode.lower()
            if mode == "tda":
                artifacts = self._run_tda(asa_data, state, log_callback)
            elif mode == "decode":
                artifacts = self._run_decode_only(asa_data, state, log_callback)
            elif mode == "cohomap":
                artifacts = self._run_cohomap(asa_data, state, log_callback)
            elif mode == "pathcompare":
                artifacts = self._run_pathcompare(asa_data, state, log_callback)
            elif mode == "cohospace":
                artifacts = self._run_cohospace(asa_data, state, log_callback)
            elif mode == "fr":
                artifacts = self._run_fr(asa_data, state, log_callback)
            elif mode == "frm":
                artifacts = self._run_frm(asa_data, state, log_callback)
            elif mode == "gridscore":
                artifacts = self._run_gridscore(asa_data, state, log_callback, progress_callback)
            elif mode == "gridscore_inspect":
                artifacts = self._run_gridscore_inspect(
                    asa_data, state, log_callback, progress_callback
                )
            else:
                raise ProcessingError(f"Unknown analysis mode: {state.analysis_mode}")

            progress_callback(100)
            elapsed = time.time() - t0

            return PipelineResult(
                success=True,
                artifacts=artifacts,
                summary=f"Completed {state.analysis_mode} analysis in {elapsed:.1f}s",
                elapsed_time=elapsed,
            )

        except Exception as e:
            elapsed = time.time() - t0
            log_callback(f"Error: {e}")
            return PipelineResult(
                success=False,
                artifacts=artifacts,
                summary=f"Failed after {elapsed:.1f}s",
                error=str(e),
                elapsed_time=elapsed,
            )

    def _load_data(self, state: WorkflowState) -> dict[str, Any]:
        if state.input_mode == "asa":
            path = resolve_path(state, state.asa_file)
            data = np.load(path, allow_pickle=True)
            return {k: data[k] for k in data.files}
        if state.input_mode == "neuron_traj":
            neuron_path = resolve_path(state, state.neuron_file)
            traj_path = resolve_path(state, state.traj_file)
            neuron_data = np.load(neuron_path, allow_pickle=True)
            traj_data = np.load(traj_path, allow_pickle=True)
            return {
                "spike": neuron_data.get("spike", neuron_data),
                "x": traj_data["x"],
                "y": traj_data["y"],
                "t": traj_data["t"],
            }
        raise ProcessingError(f"Unknown input mode: {state.input_mode}")

    def _run_tda(
        self, asa_data: dict[str, Any], state: WorkflowState, log_callback
    ) -> dict[str, Path]:
        from canns.analyzer.data.asa import TDAConfig, tda_vis
        from canns.analyzer.data.asa.tda import _plot_barcode, _plot_barcode_with_shuffle

        out_dir = self._results_dir(state) / "TDA"
        out_dir.mkdir(parents=True, exist_ok=True)

        params = state.analysis_params
        config = TDAConfig(
            dim=params.get("dim", 6),
            num_times=params.get("num_times", 5),
            active_times=params.get("active_times", 15000),
            k=params.get("k", 1000),
            n_points=params.get("n_points", 1200),
            metric=params.get("metric", "cosine"),
            nbs=params.get("nbs", 800),
            maxdim=params.get("maxdim", 1),
            coeff=params.get("coeff", 47),
            show=False,
            do_shuffle=params.get("do_shuffle", False),
            num_shuffles=params.get("num_shuffles", 1000),
            progress_bar=False,
            standardize=False,
        )

        log_callback("Computing persistent homology...")

        if self._embed_data is None:
            raise ProcessingError("No preprocessed data available. Run preprocessing first.")
        if not isinstance(self._embed_data, np.ndarray) or self._embed_data.ndim != 2:
            raise ProcessingError(
                "TDA requires a dense spike matrix (T,N). "
                "Please choose 'Embed Spike Trains' in preprocessing or provide a dense spike matrix in the .npz."
            )

        persistence_path = out_dir / "persistence.npz"
        barcode_path = out_dir / "barcode.png"

        embed_hash = self._embed_hash or self._hash_obj({"embed": "unknown"})
        tda_hash = self._hash_obj({"embed_hash": embed_hash, "params": params})

        if self._stage_cache_hit(out_dir, tda_hash, [persistence_path, barcode_path]):
            log_callback("♻️ Using cached TDA results.")
            return {"persistence": persistence_path, "barcode": barcode_path}

        embed_data = self._embed_data
        if params.get("standardize", False):
            try:
                from sklearn.preprocessing import StandardScaler

                embed_data = StandardScaler().fit_transform(embed_data)
            except Exception as e:
                raise ProcessingError(f"StandardScaler failed: {e}") from e

        result = tda_vis(
            embed_data=embed_data,
            config=config,
        )

        np.savez_compressed(persistence_path, persistence_result=result)

        try:
            persistence = result.get("persistence")
            shuffle_max = result.get("shuffle_max")
            if config.do_shuffle and shuffle_max is not None:
                fig = _plot_barcode_with_shuffle(persistence, shuffle_max)
            else:
                fig = _plot_barcode(persistence)
            fig.savefig(barcode_path, dpi=200, bbox_inches="tight")
            try:
                import matplotlib.pyplot as plt

                plt.close(fig)
            except Exception:
                pass
        except Exception as e:
            log_callback(f"Warning: failed to save barcode: {e}")

        self._write_cache_meta(
            self._stage_cache_path(out_dir),
            {"hash": tda_hash, "embed_hash": embed_hash, "params": params},
        )

        return {"persistence": persistence_path, "barcode": barcode_path}

    def _load_or_run_decode(
        self,
        asa_data: dict[str, Any],
        persistence_path: Path,
        state: WorkflowState,
        log_callback,
    ) -> dict[str, Any]:
        from canns.analyzer.data.asa import (
            decode_circular_coordinates,
            decode_circular_coordinates_multi,
        )

        decode_dir = self._results_dir(state) / "CohoMap"
        decode_dir.mkdir(parents=True, exist_ok=True)
        decode_path = decode_dir / "decoding.npz"

        params = state.analysis_params
        decode_version = str(params.get("decode_version", "v2"))
        num_circ = int(params.get("num_circ", 2))
        decode_params = {
            "real_ground": params.get("real_ground", True),
            "real_of": params.get("real_of", True),
            "decode_version": decode_version,
            "num_circ": num_circ,
        }
        persistence_hash = self._hash_file(persistence_path)
        decode_hash = self._hash_obj(
            {"persistence_hash": persistence_hash, "params": decode_params}
        )

        meta_path = self._stage_cache_path(decode_dir)
        meta = self._load_cache_meta(meta_path)
        if decode_path.exists() and meta.get("decode_hash") == decode_hash:
            log_callback("♻️ Using cached decoding.")
            return self._load_npz_dict(decode_path)

        log_callback("Decoding circular coordinates...")
        persistence_result = self._load_npz_dict(persistence_path)
        if decode_version == "v0":
            decode_result = decode_circular_coordinates(
                persistence_result=persistence_result,
                spike_data=asa_data,
                real_ground=decode_params["real_ground"],
                real_of=decode_params["real_of"],
                save_path=str(decode_path),
            )
        else:
            if self._embed_data is None:
                raise ProcessingError("No preprocessed data available for decode v2.")
            spike_data = dict(asa_data)
            spike_data["spike"] = self._embed_data
            decode_result = decode_circular_coordinates_multi(
                persistence_result=persistence_result,
                spike_data=spike_data,
                save_path=str(decode_path),
                num_circ=num_circ,
            )

        meta["decode_hash"] = decode_hash
        meta["persistence_hash"] = persistence_hash
        meta["decode_params"] = decode_params
        self._write_cache_meta(meta_path, meta)
        return decode_result

    def _run_decode_only(
        self, asa_data: dict[str, Any], state: WorkflowState, log_callback
    ) -> dict[str, Path]:
        tda_dir = self._results_dir(state) / "TDA"
        persistence_path = tda_dir / "persistence.npz"
        if not persistence_path.exists():
            raise ProcessingError("TDA results not found. Run TDA analysis first.")

        self._load_or_run_decode(asa_data, persistence_path, state, log_callback)
        return {"decoding": self._results_dir(state) / "CohoMap" / "decoding.npz"}

    def _run_cohomap(
        self, asa_data: dict[str, Any], state: WorkflowState, log_callback
    ) -> dict[str, Path]:
        from canns.analyzer.data.asa import plot_cohomap_scatter_multi
        from canns.analyzer.visualization import PlotConfigs

        tda_dir = self._results_dir(state) / "TDA"
        persistence_path = tda_dir / "persistence.npz"
        if not persistence_path.exists():
            raise ProcessingError("TDA results not found. Run TDA analysis first.")

        out_dir = self._results_dir(state) / "CohoMap"
        out_dir.mkdir(parents=True, exist_ok=True)

        decode_result = self._load_or_run_decode(asa_data, persistence_path, state, log_callback)

        params = state.analysis_params
        subsample = int(params.get("cohomap_subsample", 10))

        cohomap_path = out_dir / "cohomap.png"
        stage_hash = self._hash_obj(
            {
                "decode_hash": self._load_cache_meta(self._stage_cache_path(out_dir)).get(
                    "decode_hash"
                ),
                "plot": "cohomap",
                "subsample": subsample,
            }
        )
        if self._stage_cache_hit(out_dir, stage_hash, [cohomap_path]):
            log_callback("♻️ Using cached CohoMap plot.")
            return {"decoding": out_dir / "decoding.npz", "cohomap": cohomap_path}

        log_callback("Generating cohomology map...")
        pos = self._aligned_pos if self._aligned_pos is not None else asa_data
        config = PlotConfigs.cohomap(show=False, save_path=str(cohomap_path))
        plot_cohomap_scatter_multi(
            decoding_result=decode_result,
            position_data={"x": pos["x"], "y": pos["y"]},
            config=config,
            subsample=subsample,
        )

        self._write_cache_meta(
            self._stage_cache_path(out_dir),
            {
                **self._load_cache_meta(self._stage_cache_path(out_dir)),
                "hash": stage_hash,
            },
        )

        return {"decoding": out_dir / "decoding.npz", "cohomap": cohomap_path}

    def _run_pathcompare(
        self, asa_data: dict[str, Any], state: WorkflowState, log_callback
    ) -> dict[str, Path]:
        from canns.analyzer.data.asa import (
            align_coords_to_position_1d,
            align_coords_to_position_2d,
            apply_angle_scale,
            plot_path_compare_1d,
            plot_path_compare_2d,
        )
        from canns.analyzer.data.asa.path import (
            find_coords_matrix,
            find_times_box,
            resolve_time_slice,
        )
        from canns.analyzer.visualization import PlotConfigs

        tda_dir = self._results_dir(state) / "TDA"
        persistence_path = tda_dir / "persistence.npz"
        if not persistence_path.exists():
            raise ProcessingError("TDA results not found. Run TDA analysis first.")

        decode_result = self._load_or_run_decode(asa_data, persistence_path, state, log_callback)

        out_dir = self._results_dir(state) / "PathCompare"
        out_dir.mkdir(parents=True, exist_ok=True)

        params = state.analysis_params or {}
        pc_params = (
            params.get("pathcompare") if isinstance(params.get("pathcompare"), dict) else params
        )

        def _param(key: str, default: Any = None) -> Any:
            return pc_params.get(key, default) if isinstance(pc_params, dict) else default

        angle_scale = _param("angle_scale", _param("theta_scale", "rad"))
        dim_mode = _param("dim_mode", "2d")
        dim = int(_param("dim", 1))
        dim1 = int(_param("dim1", 1))
        dim2 = int(_param("dim2", 2))
        use_box = bool(_param("use_box", False))
        interp_full = bool(_param("interp_full", True))
        coords_key = _param("coords_key")
        times_key = _param("times_key", _param("times_box_key"))
        slice_mode = _param("slice_mode", "time")
        tmin = _param("tmin")
        tmax = _param("tmax")
        imin = _param("imin")
        imax = _param("imax")
        stride = int(_param("stride", 1))
        tail = int(_param("tail", 300))
        fps = int(_param("fps", 20))
        no_wrap = bool(_param("no_wrap", False))
        animation_format = str(_param("animation_format", "none")).lower()
        if animation_format not in {"none", "gif", "mp4"}:
            animation_format = "none"

        coords_raw, _ = find_coords_matrix(
            decode_result,
            coords_key=coords_key,
            prefer_box_fallback=use_box,
        )

        if dim_mode == "1d":
            idx = max(0, dim - 1)
            if idx >= coords_raw.shape[1]:
                raise ProcessingError(f"dim out of range for coords shape {coords_raw.shape}")
            coords1 = coords_raw[:, idx]
        else:
            idx1 = max(0, dim1 - 1)
            idx2 = max(0, dim2 - 1)
            if idx1 >= coords_raw.shape[1] or idx2 >= coords_raw.shape[1]:
                raise ProcessingError(f"dim1/dim2 out of range for coords shape {coords_raw.shape}")
            coords2 = coords_raw[:, [idx1, idx2]]

        pos = self._aligned_pos if self._aligned_pos is not None else asa_data
        t_full = np.asarray(pos["t"]).ravel()
        x_full = np.asarray(pos["x"]).ravel()
        y_full = np.asarray(pos["y"]).ravel()

        if use_box:
            if times_key:
                times_box = decode_result.get(times_key)
            else:
                times_box, _ = find_times_box(decode_result)
        else:
            times_box = None

        log_callback("Aligning decoded coordinates to position...")
        if dim_mode == "1d":
            t_use, x_use, y_use, coords_use, _ = align_coords_to_position_1d(
                t_full=t_full,
                x_full=x_full,
                y_full=y_full,
                coords1=coords1,
                use_box=use_box,
                times_box=times_box,
                interp_to_full=interp_full,
            )
        else:
            t_use, x_use, y_use, coords_use, _ = align_coords_to_position_2d(
                t_full=t_full,
                x_full=x_full,
                y_full=y_full,
                coords2=coords2,
                use_box=use_box,
                times_box=times_box,
                interp_to_full=interp_full,
            )
        scale = str(angle_scale) if str(angle_scale) in {"rad", "deg", "unit", "auto"} else "rad"
        coords_use = apply_angle_scale(coords_use, scale)
        if not no_wrap:
            coords_use = np.mod(coords_use, 2 * np.pi)

        if slice_mode == "index":
            i0, i1 = resolve_time_slice(t_use, None, None, imin, imax)
        else:
            i0, i1 = resolve_time_slice(t_use, tmin, tmax, None, None)

        stride = max(1, stride)
        idx = slice(i0, i1, stride)
        t_use = t_use[idx]
        x_use = x_use[idx]
        y_use = y_use[idx]
        coords_use = coords_use[idx]

        out_path = out_dir / "path_compare.png"
        anim_path: Path | None = None
        if animation_format == "gif":
            anim_path = out_dir / "path_compare.gif"
        elif animation_format == "mp4":
            anim_path = out_dir / "path_compare.mp4"
        decode_meta = self._load_cache_meta(
            self._stage_cache_path(self._results_dir(state) / "CohoMap")
        )
        stage_hash = self._hash_obj(
            {
                "persistence": self._hash_file(persistence_path),
                "decode_hash": decode_meta.get("decode_hash"),
                "params": {
                    "angle_scale": scale,
                    "dim_mode": dim_mode,
                    "dim": dim,
                    "dim1": dim1,
                    "dim2": dim2,
                    "use_box": use_box,
                    "interp_full": interp_full,
                    "coords_key": coords_key,
                    "times_key": times_key,
                    "slice_mode": slice_mode,
                    "tmin": tmin,
                    "tmax": tmax,
                    "imin": imin,
                    "imax": imax,
                    "stride": stride,
                    "tail": tail,
                    "fps": fps,
                    "no_wrap": no_wrap,
                    "animation_format": animation_format,
                },
            }
        )
        required = [out_path]
        if anim_path is not None:
            required.append(anim_path)
        if self._stage_cache_hit(out_dir, stage_hash, required):
            log_callback("♻️ Using cached PathCompare plot.")
            artifacts = {"path_compare": out_path}
            if anim_path is not None:
                if anim_path.suffix == ".gif":
                    artifacts["path_compare_gif"] = anim_path
                else:
                    artifacts["path_compare_mp4"] = anim_path
            return artifacts

        log_callback("Generating path comparison...")
        if dim_mode == "1d":
            config = PlotConfigs.path_compare_1d(show=False, save_path=str(out_path))
            plot_path_compare_1d(x_use, y_use, coords_use, config=config)
        else:
            config = PlotConfigs.path_compare_2d(show=False, save_path=str(out_path))
            plot_path_compare_2d(x_use, y_use, coords_use, config=config)

        artifacts: dict[str, Path] = {"path_compare": out_path}

        if anim_path is not None:
            try:
                if dim_mode == "1d":
                    self._render_pathcompare_1d_animation(
                        x_use,
                        y_use,
                        coords_use,
                        t_use,
                        anim_path,
                        tail=tail,
                        fps=fps,
                        log_callback=log_callback,
                    )
                else:
                    self._render_pathcompare_2d_animation(
                        x_use,
                        y_use,
                        coords_use,
                        t_use,
                        anim_path,
                        tail=tail,
                        fps=fps,
                        log_callback=log_callback,
                    )
                if anim_path.suffix == ".gif":
                    artifacts["path_compare_gif"] = anim_path
                else:
                    artifacts["path_compare_mp4"] = anim_path
            except Exception as e:
                log_callback(f"Warning: failed to render PathCompare animation: {e}")

        self._write_cache_meta(self._stage_cache_path(out_dir), {"hash": stage_hash})
        return artifacts

    def _render_pathcompare_1d_animation(
        self,
        x: np.ndarray,
        y: np.ndarray,
        coords: np.ndarray,
        t: np.ndarray,
        save_path: Path,
        *,
        tail: int,
        fps: int,
        log_callback,
    ) -> None:
        import matplotlib.pyplot as plt

        x = np.asarray(x).ravel()
        y = np.asarray(y).ravel()
        theta = np.asarray(coords).ravel()
        t = np.asarray(t).ravel()

        n_frames = len(theta)
        if n_frames == 0:
            raise ProcessingError("PathCompare animation has no frames.")

        # Downsample if too many frames for animation
        if n_frames > 20000:
            factor = int(np.ceil(n_frames / 20000))
            idx = np.arange(0, n_frames, factor)
            x = x[idx]
            y = y[idx]
            theta = theta[idx]
            t = t[idx]
            n_frames = len(theta)
            log_callback(f"PathCompare animation downsampled by x{factor} (frames={n_frames}).")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=120)

        ax1.set_title("Physical path")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.set_aspect("equal")
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)
        pad_x = (x_max - x_min) * 0.05 if x_max > x_min else 1.0
        pad_y = (y_max - y_min) * 0.05 if y_max > y_min else 1.0
        ax1.set_xlim(x_min - pad_x, x_max + pad_x)
        ax1.set_ylim(y_min - pad_y, y_max + pad_y)

        ax2.set_title("Decoded coho path (1D)")
        ax2.set_aspect("equal")
        ax2.axis("off")
        ax2.set_xlim(-1.2, 1.2)
        ax2.set_ylim(-1.2, 1.2)

        (phys_trail,) = ax1.plot([], [], lw=1.0)
        phys_dot = ax1.scatter([], [], s=30)
        (circ_trail,) = ax2.plot([], [], lw=1.0)
        circ_dot = ax2.scatter([], [], s=30)
        title_text = fig.suptitle("", y=1.02)

        def update(k: int) -> None:
            a0 = max(0, k - tail) if tail > 0 else 0
            xs = x[a0 : k + 1]
            ys = y[a0 : k + 1]
            phys_trail.set_data(xs, ys)
            phys_dot.set_offsets(np.array([[x[k], y[k]]]))

            x_unit = np.cos(theta[a0 : k + 1])
            y_unit = np.sin(theta[a0 : k + 1])
            circ_trail.set_data(x_unit, y_unit)
            circ_dot.set_offsets(np.array([[np.cos(theta[k]), np.sin(theta[k])]]))

            title_text.set_text(f"t = {float(t[k]):.3f}s  (frame {k + 1}/{n_frames})")

        fig.tight_layout()
        self._save_animation(fig, update, n_frames, save_path, fps, log_callback)
        plt.close(fig)

    def _render_pathcompare_2d_animation(
        self,
        x: np.ndarray,
        y: np.ndarray,
        coords: np.ndarray,
        t: np.ndarray,
        save_path: Path,
        *,
        tail: int,
        fps: int,
        log_callback,
    ) -> None:
        import matplotlib.pyplot as plt

        from canns.analyzer.data.asa.path import (
            draw_base_parallelogram,
            skew_transform,
            snake_wrap_trail_in_parallelogram,
        )

        x = np.asarray(x).ravel()
        y = np.asarray(y).ravel()
        coords = np.asarray(coords)
        t = np.asarray(t).ravel()

        if coords.ndim != 2 or coords.shape[1] < 2:
            raise ProcessingError("PathCompare 2D animation requires coords with 2 columns.")

        n_frames = len(coords)
        if n_frames == 0:
            raise ProcessingError("PathCompare animation has no frames.")

        if n_frames > 20000:
            factor = int(np.ceil(n_frames / 20000))
            idx = np.arange(0, n_frames, factor)
            x = x[idx]
            y = y[idx]
            coords = coords[idx]
            t = t[idx]
            n_frames = len(coords)
            log_callback(f"PathCompare animation downsampled by x{factor} (frames={n_frames}).")

        xy_skew = skew_transform(coords[:, :2])

        e1 = np.array([2 * np.pi, 0.0])
        e2 = np.array([np.pi, np.sqrt(3) * np.pi])
        corners = np.vstack([[0.0, 0.0], e1, e2, e1 + e2])
        xm, ym = corners.min(axis=0)
        xM, yM = corners.max(axis=0)
        px2 = 0.05 * (xM - xm + 1e-9)
        py2 = 0.05 * (yM - ym + 1e-9)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=120)

        ax1.set_title("Physical path")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.set_aspect("equal")
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)
        pad_x = 0.05 * (x_max - x_min + 1e-9)
        pad_y = 0.05 * (y_max - y_min + 1e-9)
        ax1.set_xlim(x_min - pad_x, x_max + pad_x)
        ax1.set_ylim(y_min - pad_y, y_max + pad_y)

        ax2.set_title("Torus path (skew)")
        ax2.set_xlabel(r"$\theta_1 + \frac{1}{2}\theta_2$")
        ax2.set_ylabel(r"$\frac{\sqrt{3}}{2}\theta_2$")
        ax2.set_aspect("equal")
        draw_base_parallelogram(ax2)
        ax2.set_xlim(xm - px2, xM + px2)
        ax2.set_ylim(ym - py2, yM + py2)

        (phys_trail,) = ax1.plot([], [], lw=1.0)
        phys_dot = ax1.scatter([], [], s=30)
        (tor_trail,) = ax2.plot([], [], lw=1.0)
        tor_dot = ax2.scatter([], [], s=30)
        title_text = fig.suptitle("", y=1.02)

        def update(k: int) -> None:
            a0 = max(0, k - tail) if tail > 0 else 0
            xs = x[a0 : k + 1]
            ys = y[a0 : k + 1]
            phys_trail.set_data(xs, ys)
            phys_dot.set_offsets(np.array([[x[k], y[k]]]))

            seg = snake_wrap_trail_in_parallelogram(xy_skew[a0 : k + 1], e1=e1, e2=e2)
            tor_trail.set_data(seg[:, 0], seg[:, 1])
            tor_dot.set_offsets(np.array([[xy_skew[k, 0], xy_skew[k, 1]]]))

            title_text.set_text(f"t = {float(t[k]):.3f}s  (frame {k + 1}/{n_frames})")

        fig.tight_layout()
        self._save_animation(fig, update, n_frames, save_path, fps, log_callback)
        plt.close(fig)

    def _save_animation(
        self,
        fig,
        update_func,
        n_frames: int,
        save_path: Path,
        fps: int,
        log_callback,
    ) -> None:
        if save_path.suffix.lower() == ".gif":
            from matplotlib.animation import FuncAnimation, PillowWriter

            def _update(k: int):
                update_func(k)
                return []

            interval_ms = int(1000 / max(1, fps))
            ani = FuncAnimation(fig, _update, frames=n_frames, interval=interval_ms, blit=True)

            last_pct = {"v": -1}

            def _progress(i: int, total: int) -> None:
                if not total:
                    return
                pct = int((i + 1) * 100 / total)
                if pct != last_pct["v"]:
                    last_pct["v"] = pct
                    log_callback(f"__PCANIM__ {pct} {i + 1}/{total}")

            ani.save(
                str(save_path),
                writer=PillowWriter(fps=fps),
                progress_callback=_progress,
            )
            return

        from canns.analyzer.visualization.core import (
            get_imageio_writer_kwargs,
            get_matplotlib_writer,
            select_animation_backend,
        )

        backend_selection = select_animation_backend(
            save_path=str(save_path),
            requested_backend="auto",
            check_imageio_plugins=True,
        )
        for warning in backend_selection.warnings:
            log_callback(f"⚠️ {warning}")

        if backend_selection.backend == "imageio":
            import imageio

            writer_kwargs, mode = get_imageio_writer_kwargs(str(save_path), fps)
            last_pct = -1
            with imageio.get_writer(str(save_path), mode=mode, **writer_kwargs) as writer:
                for k in range(n_frames):
                    update_func(k)
                    fig.canvas.draw()
                    frame = np.asarray(fig.canvas.buffer_rgba())
                    writer.append_data(frame[:, :, :3])
                    pct = int((k + 1) * 100 / n_frames)
                    if pct != last_pct:
                        last_pct = pct
                        log_callback(f"__PCANIM__ {pct} {k + 1}/{n_frames}")
            return

        from matplotlib.animation import FuncAnimation

        def _update(k: int):
            update_func(k)
            return []

        interval_ms = int(1000 / max(1, fps))
        ani = FuncAnimation(fig, _update, frames=n_frames, interval=interval_ms, blit=False)
        writer = get_matplotlib_writer(str(save_path), fps=fps)
        ani.save(str(save_path), writer=writer)

    def _run_cohospace(
        self, asa_data: dict[str, Any], state: WorkflowState, log_callback
    ) -> dict[str, Path]:
        from canns.analyzer.data.asa import (
            plot_cohospace_scatter_neuron_1d,
            plot_cohospace_scatter_neuron_2d,
            plot_cohospace_scatter_population_1d,
            plot_cohospace_scatter_population_2d,
            plot_cohospace_scatter_trajectory_1d,
            plot_cohospace_scatter_trajectory_2d,
        )
        from canns.analyzer.data.asa.cohospace_scatter import (
            compute_cohoscore_scatter_1d,
            compute_cohoscore_scatter_2d,
            plot_cohospace_scatter_neuron_skewed,
            plot_cohospace_scatter_population_skewed,
        )
        from canns.analyzer.visualization import PlotConfigs

        tda_dir = self._results_dir(state) / "TDA"
        persistence_path = tda_dir / "persistence.npz"
        if not persistence_path.exists():
            raise ProcessingError("TDA results not found. Run TDA analysis first.")

        decode_result = self._load_or_run_decode(asa_data, persistence_path, state, log_callback)

        out_dir = self._results_dir(state) / "CohoSpace"
        out_dir.mkdir(parents=True, exist_ok=True)

        params = state.analysis_params
        artifacts: dict[str, Path] = {}

        coords = np.asarray(decode_result.get("coords"))
        coordsbox = np.asarray(decode_result.get("coordsbox"))
        if coords.ndim != 2 or coords.shape[1] < 1:
            raise ProcessingError("decode_result['coords'] must be 2D.")

        dim_mode = str(params.get("dim_mode", "2d"))
        dim = int(params.get("dim", 1))
        dim1 = int(params.get("dim1", 1))
        dim2 = int(params.get("dim2", 2))
        mode = str(params.get("mode", "fr"))
        top_percent = float(params.get("top_percent", 5.0))
        view = str(params.get("view", "both"))
        subsample = int(params.get("subsample", 2))
        unfold = str(params.get("unfold", "square"))
        skew_show_grid = bool(params.get("skew_show_grid", True))
        skew_tiles = int(params.get("skew_tiles", 0))
        enable_score = bool(params.get("enable_score", True))
        top_k = int(params.get("top_k", 10))
        use_best = bool(params.get("use_best", True))
        times = decode_result.get("times")

        def pick_coords(arr: np.ndarray) -> np.ndarray:
            if dim_mode == "1d":
                idx = max(0, dim - 1)
                if idx >= arr.shape[1]:
                    raise ProcessingError(f"dim out of range for coords shape {arr.shape}")
                return arr[:, idx]
            idx1 = max(0, dim1 - 1)
            idx2 = max(0, dim2 - 1)
            if idx1 >= arr.shape[1] or idx2 >= arr.shape[1]:
                raise ProcessingError(f"dim1/dim2 out of range for coords shape {arr.shape}")
            return arr[:, [idx1, idx2]]

        coords2 = pick_coords(coords)
        coordsbox2 = pick_coords(coordsbox) if coordsbox.ndim == 2 else coords2

        if mode == "spike":
            activity = self._build_spike_matrix_from_events(asa_data)
        else:
            activity = (
                self._embed_data
                if self._embed_data is not None
                else self._build_spike_matrix_from_events(asa_data)
            )

        scores = None
        top_ids = None
        neuron_id = int(params.get("neuron_id", 0))
        if enable_score:
            try:
                if dim_mode == "1d":
                    scores = compute_cohoscore_scatter_1d(
                        coords2, activity, top_percent=top_percent, times=times
                    )
                else:
                    scores = compute_cohoscore_scatter_2d(
                        coords2, activity, top_percent=top_percent, times=times
                    )
                cohoscore_path = out_dir / "cohoscore.npy"
                np.save(cohoscore_path, scores)
            except Exception as e:
                log_callback(f"⚠️ CohoScore computation failed: {e}")
                scores = None

            if scores is not None:
                valid = np.where(~np.isnan(scores))[0]
                if valid.size > 0:
                    sorted_idx = valid[np.argsort(scores[valid])]
                    top_ids = sorted_idx[: min(top_k, len(sorted_idx))]
                    top_ids_path = out_dir / "cohospace_top_ids.npy"
                    np.save(top_ids_path, top_ids)
                else:
                    log_callback("⚠️ CohoScore: all values are NaN.")

        if view in {"both", "single"} and enable_score and scores is not None and use_best:
            valid = np.where(~np.isnan(scores))[0]
            if valid.size > 0:
                best_id = int(valid[np.argmin(scores[valid])])
                neuron_id = best_id
                log_callback(f"🎯 CohoSpace neuron auto-selected by best CohoScore: {neuron_id}")

        decode_meta = self._load_cache_meta(
            self._stage_cache_path(self._results_dir(state) / "CohoMap")
        )
        stage_hash = self._hash_obj(
            {
                "persistence": self._hash_file(persistence_path),
                "decode_hash": decode_meta.get("decode_hash"),
                "params": params,
            }
        )
        meta_path = self._stage_cache_path(out_dir)
        required = [out_dir / "cohospace_trajectory.png"]
        if view in {"both", "population"}:
            required.append(out_dir / "cohospace_population.png")
        if view in {"both", "single"}:
            required.append(out_dir / f"cohospace_neuron_{neuron_id}.png")

        if self._stage_cache_hit(out_dir, stage_hash, required):
            log_callback("♻️ Using cached CohoSpace plots.")
            artifacts = {"trajectory": out_dir / "cohospace_trajectory.png"}
            if view in {"both", "single"}:
                artifacts["neuron"] = out_dir / f"cohospace_neuron_{neuron_id}.png"
            if view in {"both", "population"}:
                artifacts["population"] = out_dir / "cohospace_population.png"
            return artifacts

        log_callback("Plotting cohomology space trajectory...")
        traj_path = out_dir / "cohospace_trajectory.png"
        if dim_mode == "1d":
            traj_cfg = PlotConfigs.cohospace_trajectory_1d(show=False, save_path=str(traj_path))
            plot_cohospace_scatter_trajectory_1d(
                coords=coords2,
                times=None,
                subsample=subsample,
                config=traj_cfg,
            )
        else:
            traj_cfg = PlotConfigs.cohospace_trajectory_2d(show=False, save_path=str(traj_path))
            plot_cohospace_scatter_trajectory_2d(
                coords=coords2,
                times=None,
                subsample=subsample,
                config=traj_cfg,
            )
        artifacts["trajectory"] = traj_path

        if neuron_id is not None and view in {"both", "single"}:
            log_callback(f"Plotting neuron {neuron_id}...")
            neuron_path = out_dir / f"cohospace_neuron_{neuron_id}.png"
            if unfold == "skew" and dim_mode != "1d":
                plot_cohospace_scatter_neuron_skewed(
                    coords=coordsbox2,
                    activity=activity,
                    neuron_id=int(neuron_id),
                    mode=mode,
                    top_percent=top_percent,
                    times=times,
                    save_path=str(neuron_path),
                    show=False,
                    show_grid=skew_show_grid,
                    n_tiles=skew_tiles,
                )
            else:
                if dim_mode == "1d":
                    neuron_cfg = PlotConfigs.cohospace_neuron_1d(
                        show=False, save_path=str(neuron_path)
                    )
                    plot_cohospace_scatter_neuron_1d(
                        coords=coordsbox2,
                        activity=activity,
                        neuron_id=int(neuron_id),
                        mode=mode,
                        top_percent=top_percent,
                        times=times,
                        config=neuron_cfg,
                    )
                else:
                    neuron_cfg = PlotConfigs.cohospace_neuron_2d(
                        show=False, save_path=str(neuron_path)
                    )
                    plot_cohospace_scatter_neuron_2d(
                        coords=coordsbox2,
                        activity=activity,
                        neuron_id=int(neuron_id),
                        mode=mode,
                        top_percent=top_percent,
                        times=times,
                        config=neuron_cfg,
                    )
            artifacts["neuron"] = neuron_path

        if view in {"both", "population"}:
            log_callback("Plotting population activity...")
            pop_path = out_dir / "cohospace_population.png"
            if enable_score and top_ids is not None:
                neuron_ids = [int(i) for i in top_ids.tolist()]
                log_callback(f"CohoSpace: aggregating top-{len(neuron_ids)} neurons by CohoScore.")
            else:
                neuron_ids = list(range(activity.shape[1]))
            if unfold == "skew" and dim_mode != "1d":
                plot_cohospace_scatter_population_skewed(
                    coords=coords2,
                    activity=activity,
                    neuron_ids=neuron_ids,
                    mode=mode,
                    top_percent=top_percent,
                    times=times,
                    save_path=str(pop_path),
                    show=False,
                    show_grid=skew_show_grid,
                    n_tiles=skew_tiles,
                )
            else:
                if dim_mode == "1d":
                    pop_cfg = PlotConfigs.cohospace_population_1d(
                        show=False, save_path=str(pop_path)
                    )
                    plot_cohospace_scatter_population_1d(
                        coords=coords2,
                        activity=activity,
                        neuron_ids=neuron_ids,
                        mode=mode,
                        top_percent=top_percent,
                        times=times,
                        config=pop_cfg,
                    )
                else:
                    pop_cfg = PlotConfigs.cohospace_population_2d(
                        show=False, save_path=str(pop_path)
                    )
                    plot_cohospace_scatter_population_2d(
                        coords=coords2,
                        activity=activity,
                        neuron_ids=neuron_ids,
                        mode=mode,
                        top_percent=top_percent,
                        times=times,
                        config=pop_cfg,
                    )
            artifacts["population"] = pop_path

        self._write_cache_meta(meta_path, {"hash": stage_hash})
        return artifacts

    def _run_fr(
        self, asa_data: dict[str, Any], state: WorkflowState, log_callback
    ) -> dict[str, Path]:
        from canns.analyzer.data.asa import compute_fr_heatmap_matrix, save_fr_heatmap_png
        from canns.analyzer.visualization import PlotConfigs

        out_dir = self._results_dir(state) / "FR"
        out_dir.mkdir(parents=True, exist_ok=True)

        params = state.analysis_params
        neuron_range = params.get("neuron_range", None)
        time_range = params.get("time_range", None)
        normalize = params.get("normalize", "zscore_per_neuron")
        if normalize in {"none", "", None}:
            normalize = None
        mode = params.get("mode", "fr")

        if mode == "spike":
            spike_data = self._build_spike_matrix_from_events(asa_data)
        else:
            spike_data = self._embed_data

        if spike_data is None:
            raise ProcessingError("No spike data available for FR.")

        out_path = out_dir / "fr_heatmap.png"
        stage_hash = self._hash_obj(
            {
                "embed_hash": self._embed_hash,
                "params": params,
            }
        )
        if self._stage_cache_hit(out_dir, stage_hash, [out_path]):
            log_callback("♻️ Using cached FR heatmap.")
            return {"fr_heatmap": out_path}

        log_callback("Computing firing rate heatmap...")
        fr_matrix = compute_fr_heatmap_matrix(
            spike_data,
            neuron_range=neuron_range,
            time_range=time_range,
            normalize=normalize,
        )

        config = PlotConfigs.fr_heatmap(show=False, save_path=str(out_path))
        save_fr_heatmap_png(fr_matrix, config=config, dpi=200)

        self._write_cache_meta(self._stage_cache_path(out_dir), {"hash": stage_hash})
        return {"fr_heatmap": out_path}

    def _run_frm(
        self, asa_data: dict[str, Any], state: WorkflowState, log_callback
    ) -> dict[str, Path]:
        from canns.analyzer.data.asa import compute_frm, plot_frm
        from canns.analyzer.visualization import PlotConfigs

        out_dir = self._results_dir(state) / "FRM"
        out_dir.mkdir(parents=True, exist_ok=True)

        params = state.analysis_params
        neuron_id = int(params.get("neuron_id", 0))
        bins = int(params.get("bin_size", 50))
        min_occupancy = int(params.get("min_occupancy", 1))
        smoothing = bool(params.get("smoothing", False))
        smooth_sigma = float(params.get("smooth_sigma", 2.0))
        mode = str(params.get("mode", "fr"))

        if mode == "spike":
            spike_data = self._build_spike_matrix_from_events(asa_data)
        else:
            spike_data = self._embed_data
        if spike_data is None:
            raise ProcessingError("No spike data available for FRM.")

        pos = self._aligned_pos if self._aligned_pos is not None else asa_data
        x = np.asarray(pos.get("x"))
        y = np.asarray(pos.get("y"))

        if x is None or y is None:
            raise ProcessingError("Position data (x,y) is required for FRM.")

        out_path = out_dir / f"frm_neuron_{neuron_id}.png"
        stage_hash = self._hash_obj(
            {
                "embed_hash": self._embed_hash,
                "params": params,
            }
        )
        if self._stage_cache_hit(out_dir, stage_hash, [out_path]):
            log_callback("♻️ Using cached FRM.")
            return {"frm": out_path}

        log_callback(f"Computing firing rate map for neuron {neuron_id}...")
        frm_result = compute_frm(
            spike_data,
            x,
            y,
            neuron_id=neuron_id,
            bins=max(1, bins),
            min_occupancy=min_occupancy,
            smoothing=smoothing,
            sigma=smooth_sigma,
            nan_for_empty=True,
        )

        config = PlotConfigs.frm(show=False, save_path=str(out_path))
        plot_frm(frm_result.frm, config=config, dpi=200)

        self._write_cache_meta(self._stage_cache_path(out_dir), {"hash": stage_hash})
        return {"frm": out_path}

    def _run_gridscore(
        self,
        asa_data: dict[str, Any],
        state: WorkflowState,
        log_callback,
        progress_callback: Callable[[int], None],
    ) -> dict[str, Path]:
        """Run batch gridness score computation."""
        import csv

        from canns.analyzer.data.cell_classification import (
            GridnessAnalyzer,
            compute_2d_autocorrelation,
        )

        params = state.analysis_params or {}
        gs_params = params.get("gridscore") if isinstance(params.get("gridscore"), dict) else {}

        def _param(key: str, default: Any) -> Any:
            if key in params:
                return params.get(key, default)
            if gs_params and key in gs_params:
                return gs_params.get(key, default)
            return default

        n_start = int(_param("neuron_start", 0))
        n_end = int(_param("neuron_end", 0))
        bins = int(_param("bins", 50))
        min_occ = int(_param("min_occupancy", 1))
        smoothing = bool(_param("smoothing", False))
        sigma = float(_param("sigma", 1.0))
        overlap = float(_param("overlap", 0.8))
        mode = str(_param("mode", "fr")).strip().lower()
        score_thr = float(_param("score_thr", 0.3))

        if mode not in {"fr", "spike"}:
            mode = "fr"
        bins = max(5, bins)
        overlap = max(0.1, min(1.0, overlap))

        pos = self._aligned_pos if self._aligned_pos is not None else asa_data
        if "x" not in pos or "y" not in pos or pos["x"] is None or pos["y"] is None:
            raise ProcessingError("GridScore requires position data (x, y).")

        if mode == "fr":
            if isinstance(self._embed_data, np.ndarray) and self._embed_data.ndim == 2:
                activity_full = self._embed_data
                log_callback("GridScore[FR]: using preprocessed spike matrix.")
            else:
                log_callback("GridScore[FR]: no preprocessed matrix, falling back to spike mode.")
                activity_full = self._build_spike_matrix_from_events(asa_data)
        else:
            activity_full = self._build_spike_matrix_from_events(asa_data)
            log_callback("GridScore[spike]: using event-based spike matrix.")

        sp = np.asarray(activity_full)
        if sp.ndim != 2:
            raise ProcessingError(f"GridScore expects 2D spike matrix, got ndim={sp.ndim}.")

        x = np.asarray(pos["x"]).ravel()
        y = np.asarray(pos["y"]).ravel()
        m = min(len(x), len(y), sp.shape[0])
        x = x[:m]
        y = y[:m]
        sp = sp[:m, :]

        finite = np.isfinite(x) & np.isfinite(y)
        if not np.all(finite):
            x = x[finite]
            y = y[finite]
            sp = sp[finite, :]

        total_neurons = sp.shape[1]
        if n_end <= 0 or n_end > total_neurons:
            n_end = total_neurons
        n_start = max(0, min(n_start, total_neurons - 1))
        n_end = max(n_start + 1, n_end)

        xmin, xmax = float(np.min(x)), float(np.max(x))
        ymin, ymax = float(np.min(y)), float(np.max(y))
        eps = 1e-12
        if xmax - xmin < eps:
            xmax = xmin + 1.0
        if ymax - ymin < eps:
            ymax = ymin + 1.0

        ix = np.floor((x - xmin) / (xmax - xmin + eps) * bins).astype(int)
        iy = np.floor((y - ymin) / (ymax - ymin + eps) * bins).astype(int)
        ix = np.clip(ix, 0, bins - 1)
        iy = np.clip(iy, 0, bins - 1)
        flat = (iy * bins + ix).astype(int)

        occ = np.bincount(flat, minlength=bins * bins).astype(float).reshape(bins, bins)
        occ_mask = occ >= float(min_occ)

        gaussian_filter = None
        if smoothing and sigma > 0:
            try:
                from scipy.ndimage import gaussian_filter as _gaussian_filter
            except Exception as e:  # pragma: no cover - optional dependency
                raise ProcessingError(f"GridScore requires scipy for smoothing: {e}") from e
            gaussian_filter = _gaussian_filter

        def _rate_map_for_neuron(col: int) -> np.ndarray:
            weights = sp[:, col].astype(float, copy=False)
            spike_map = (
                np.bincount(flat, weights=weights, minlength=bins * bins)
                .astype(float)
                .reshape(bins, bins)
            )
            rate_map = np.zeros_like(spike_map)
            rate_map[occ_mask] = spike_map[occ_mask] / occ[occ_mask]
            if gaussian_filter is not None:
                rate_map = gaussian_filter(rate_map, sigma=float(sigma), mode="nearest")
            return rate_map

        analyzer = GridnessAnalyzer()
        n_sel = n_end - n_start
        scores = np.full((n_sel,), np.nan, dtype=float)
        spacing = np.full((n_sel, 3), np.nan, dtype=float)
        orientation = np.full((n_sel, 3), np.nan, dtype=float)
        ellipse = np.full((n_sel, 5), np.nan, dtype=float)
        ellipse_theta_deg = np.full((n_sel,), np.nan, dtype=float)
        center_radius = np.full((n_sel,), np.nan, dtype=float)
        optimal_radius = np.full((n_sel,), np.nan, dtype=float)

        log_callback(
            f"GridScore: computing neurons [{n_start}, {n_end}) with bins={bins}, overlap={overlap:.2f}."
        )

        for j, nid in enumerate(range(n_start, n_end)):
            rate_map = _rate_map_for_neuron(nid)
            autocorr = compute_2d_autocorrelation(rate_map, overlap=overlap)
            result = analyzer.compute_gridness_score(autocorr)

            scores[j] = float(result.score)
            if result.spacing is not None and np.size(result.spacing) >= 3:
                spacing[j, :] = np.asarray(result.spacing).ravel()[:3]
            if result.orientation is not None and np.size(result.orientation) >= 3:
                orientation[j, :] = np.asarray(result.orientation).ravel()[:3]
            if result.ellipse is not None and np.size(result.ellipse) >= 5:
                ellipse[j, :] = np.asarray(result.ellipse).ravel()[:5]
            ellipse_theta_deg[j] = float(result.ellipse_theta_deg)
            center_radius[j] = float(result.center_radius)
            optimal_radius[j] = float(result.optimal_radius)

            if (j + 1) % max(1, n_sel // 10) == 0:
                progress = 60 + int(35 * (j + 1) / max(1, n_sel))
                progress_callback(min(98, progress))

        out_dir = self._results_dir(state) / "GRIDScore"
        out_dir.mkdir(parents=True, exist_ok=True)

        gridscore_npz = out_dir / "gridscore.npz"
        np.savez_compressed(
            str(gridscore_npz),
            neuron_start=n_start,
            neuron_end=n_end,
            neuron_ids=np.arange(n_start, n_end, dtype=int),
            bins=bins,
            min_occupancy=min_occ,
            smoothing=smoothing,
            sigma=sigma,
            overlap=overlap,
            mode=mode,
            score=scores,
            grid_score=scores,
            spacing=spacing,
            orientation=orientation,
            ellipse=ellipse,
            ellipse_theta_deg=ellipse_theta_deg,
            center_radius=center_radius,
            optimal_radius=optimal_radius,
        )

        gridscore_csv = out_dir / "gridscore.csv"
        with gridscore_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "neuron_id",
                    "grid_score",
                    "spacing1",
                    "spacing2",
                    "spacing3",
                    "orient1_deg",
                    "orient2_deg",
                    "orient3_deg",
                    "ellipse_cx",
                    "ellipse_cy",
                    "ellipse_rx",
                    "ellipse_ry",
                    "ellipse_theta_deg",
                    "center_radius",
                    "optimal_radius",
                ]
            )
            for j, nid in enumerate(range(n_start, n_end)):
                writer.writerow(
                    [
                        nid,
                        scores[j],
                        spacing[j, 0],
                        spacing[j, 1],
                        spacing[j, 2],
                        orientation[j, 0],
                        orientation[j, 1],
                        orientation[j, 2],
                        ellipse[j, 0],
                        ellipse[j, 1],
                        ellipse[j, 2],
                        ellipse[j, 3],
                        ellipse_theta_deg[j],
                        center_radius[j],
                        optimal_radius[j],
                    ]
                )

        gridscore_png = out_dir / "gridscore_summary.png"
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(1, 1, figsize=(10, 3.5))
            valid = scores[np.isfinite(scores)]
            ax.hist(valid, bins=30)
            ax.axvline(score_thr, linestyle="--")
            ax.set_title("Grid score distribution")
            ax.set_xlabel("grid score")
            ax.set_ylabel("count")
            fig.tight_layout()
            fig.savefig(gridscore_png, dpi=200)
            plt.close(fig)
        except Exception as e:
            log_callback(f"Warning: failed to save GridScore summary png: {e}")

        log_callback(f"GridScore done. Saved: {gridscore_npz} , {gridscore_csv}")
        return {
            "gridscore_npz": gridscore_npz,
            "gridscore_csv": gridscore_csv,
            "gridscore_png": gridscore_png,
        }

    def _run_gridscore_inspect(
        self,
        asa_data: dict[str, Any],
        state: WorkflowState,
        log_callback,
        progress_callback: Callable[[int], None],
    ) -> dict[str, Path]:
        """Run single-neuron GridScore inspection."""
        from canns.analyzer.data.cell_classification import (
            GridnessAnalyzer,
            compute_2d_autocorrelation,
            plot_gridness_analysis,
        )

        params = state.analysis_params or {}
        gs_params = params.get("gridscore") if isinstance(params.get("gridscore"), dict) else {}

        def _param(key: str, default: Any) -> Any:
            if key in params:
                return params.get(key, default)
            if gs_params and key in gs_params:
                return gs_params.get(key, default)
            return default

        neuron_id = int(_param("neuron_id", _param("neuron", 0)))
        bins = int(_param("bins", 50))
        min_occ = int(_param("min_occupancy", 1))
        smoothing = bool(_param("smoothing", False))
        sigma = float(_param("sigma", 1.0))
        overlap = float(_param("overlap", 0.8))
        mode = str(_param("mode", "fr")).strip().lower()

        if mode not in {"fr", "spike"}:
            mode = "fr"
        bins = max(5, bins)
        overlap = max(0.1, min(1.0, overlap))

        pos = self._aligned_pos if self._aligned_pos is not None else asa_data
        if "x" not in pos or "y" not in pos or pos["x"] is None or pos["y"] is None:
            raise ProcessingError("GridScore inspect requires position data (x, y).")

        if mode == "fr":
            if isinstance(self._embed_data, np.ndarray) and self._embed_data.ndim == 2:
                activity_full = self._embed_data
                log_callback("GridScoreInspect[FR]: using preprocessed spike matrix.")
            else:
                log_callback(
                    "GridScoreInspect[FR]: no preprocessed matrix, falling back to spike mode."
                )
                activity_full = self._build_spike_matrix_from_events(asa_data)
        else:
            activity_full = self._build_spike_matrix_from_events(asa_data)
            log_callback("GridScoreInspect[spike]: using event-based spike matrix.")

        sp = np.asarray(activity_full)
        if sp.ndim != 2:
            raise ProcessingError(f"GridScore inspect expects 2D spike matrix, got ndim={sp.ndim}.")

        x = np.asarray(pos["x"]).ravel()
        y = np.asarray(pos["y"]).ravel()
        m = min(len(x), len(y), sp.shape[0])
        x = x[:m]
        y = y[:m]
        sp = sp[:m, :]

        finite = np.isfinite(x) & np.isfinite(y)
        if not np.all(finite):
            x = x[finite]
            y = y[finite]
            sp = sp[finite, :]

        total_neurons = sp.shape[1]
        neuron_id = max(0, min(int(neuron_id), total_neurons - 1))

        xmin, xmax = float(np.min(x)), float(np.max(x))
        ymin, ymax = float(np.min(y)), float(np.max(y))
        eps = 1e-12
        if xmax - xmin < eps:
            xmax = xmin + 1.0
        if ymax - ymin < eps:
            ymax = ymin + 1.0

        ix = np.floor((x - xmin) / (xmax - xmin + eps) * bins).astype(int)
        iy = np.floor((y - ymin) / (ymax - ymin + eps) * bins).astype(int)
        ix = np.clip(ix, 0, bins - 1)
        iy = np.clip(iy, 0, bins - 1)
        flat = (iy * bins + ix).astype(int)

        occ = np.bincount(flat, minlength=bins * bins).astype(float).reshape(bins, bins)
        occ_mask = occ >= float(min_occ)

        weights = sp[:, neuron_id].astype(float, copy=False)
        spike_map = (
            np.bincount(flat, weights=weights, minlength=bins * bins)
            .astype(float)
            .reshape(bins, bins)
        )
        rate_map = np.zeros_like(spike_map)
        rate_map[occ_mask] = spike_map[occ_mask] / occ[occ_mask]
        if smoothing and sigma > 0:
            try:
                from scipy.ndimage import gaussian_filter as _gaussian_filter
            except Exception as e:  # pragma: no cover - optional dependency
                raise ProcessingError(f"GridScore requires scipy for smoothing: {e}") from e
            rate_map = _gaussian_filter(rate_map, sigma=float(sigma), mode="nearest")

        autocorr = compute_2d_autocorrelation(rate_map, overlap=overlap)
        analyzer = GridnessAnalyzer()
        result = analyzer.compute_gridness_score(autocorr)

        out_dir = self._results_dir(state) / "GRIDScore"
        out_dir.mkdir(parents=True, exist_ok=True)

        gridscore_neuron_npz = out_dir / f"gridscore_neuron_{neuron_id}.npz"
        np.savez_compressed(
            str(gridscore_neuron_npz),
            neuron_id=neuron_id,
            bins=bins,
            min_occupancy=min_occ,
            smoothing=smoothing,
            sigma=sigma,
            overlap=overlap,
            mode=mode,
            grid_score=float(result.score),
            spacing=np.asarray(result.spacing)
            if result.spacing is not None
            else np.full((3,), np.nan),
            orientation=np.asarray(result.orientation)
            if result.orientation is not None
            else np.full((3,), np.nan),
            ellipse=np.asarray(result.ellipse)
            if result.ellipse is not None
            else np.full((5,), np.nan),
            ellipse_theta_deg=float(getattr(result, "ellipse_theta_deg", np.nan)),
            center_radius=float(getattr(result, "center_radius", np.nan)),
            optimal_radius=float(getattr(result, "optimal_radius", np.nan)),
        )

        gridscore_neuron_png = out_dir / f"gridscore_neuron_{neuron_id}.png"
        plot_gridness_analysis(
            rate_map=rate_map,
            autocorr=autocorr,
            result=result,
            save_path=str(gridscore_neuron_png),
            show=False,
            title=f"GridScore neuron {neuron_id}",
        )
        progress_callback(100)

        return {
            "gridscore_neuron_npz": gridscore_neuron_npz,
            "gridscore_neuron_png": gridscore_neuron_png,
        }
