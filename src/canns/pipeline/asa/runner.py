"""Pipeline execution wrapper for ASA TUI.

This module provides async pipeline execution that integrates with the existing
canns.analyzer.data.asa module. It wraps the analysis functions and provides
progress callbacks for the TUI.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

from .state import WorkflowState, resolve_path


@dataclass
class PipelineResult:
    """Result from pipeline execution."""

    success: bool
    artifacts: Dict[str, Path]
    summary: str
    error: Optional[str] = None
    elapsed_time: float = 0.0


class ProcessingError(RuntimeError):
    """Raised when a pipeline stage fails."""
    pass


class PipelineRunner:
    """Async pipeline execution wrapper."""

    def __init__(self):
        """Initialize pipeline runner."""
        self._asa_data: Optional[Dict[str, Any]] = None
        self._embed_data: Optional[np.ndarray] = None  # Preprocessed data
        self._aligned_pos: Optional[Dict[str, np.ndarray]] = None
        self._input_hash: Optional[str] = None
        self._embed_hash: Optional[str] = None

    def has_preprocessed_data(self) -> bool:
        """Check if preprocessing has been completed."""
        return self._embed_data is not None

    def _json_safe(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable structures."""
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
        """Compute md5 hash for a file."""
        md5 = hashlib.md5()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                md5.update(chunk)
        return md5.hexdigest()

    def _hash_obj(self, obj: Any) -> str:
        payload = json.dumps(self._json_safe(obj), sort_keys=True, ensure_ascii=True).encode("utf-8")
        return self._hash_bytes(payload)

    def _cache_dir(self, state: WorkflowState) -> Path:
        return state.workdir / ".asa_cache"

    def _stage_cache_path(self, stage_dir: Path) -> Path:
        return stage_dir / "cache.json"

    def _load_cache_meta(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _write_cache_meta(self, path: Path, payload: Dict[str, Any]) -> None:
        path.write_text(json.dumps(self._json_safe(payload), ensure_ascii=True, indent=2), encoding="utf-8")

    def _stage_cache_hit(self, stage_dir: Path, expected_hash: str, required_files: list[Path]) -> bool:
        if not all(p.exists() for p in required_files):
            return False
        meta = self._load_cache_meta(self._stage_cache_path(stage_dir))
        return meta.get("hash") == expected_hash

    def _compute_input_hash(self, state: WorkflowState) -> str:
        """Compute md5 hash for input data files."""
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

    def _load_npz_dict(self, path: Path) -> Dict[str, Any]:
        """Load npz into a dict, handling wrapped dict entries."""
        data = np.load(path, allow_pickle=True)
        for key in ("persistence_result", "decode_result"):
            if key in data.files:
                return data[key].item()
        return {k: data[k] for k in data.files}

    async def run_preprocessing(
        self,
        state: WorkflowState,
        log_callback: Callable[[str], None],
        progress_callback: Callable[[int], None],
    ) -> PipelineResult:
        """Run preprocessing pipeline to generate embed_data.

        Args:
            state: Current workflow state
            log_callback: Callback for log messages
            progress_callback: Callback for progress updates (0-100)

        Returns:
            PipelineResult with preprocessing status
        """
        t0 = time.time()

        try:
            # Stage 1: Load data
            log_callback("Loading data...")
            progress_callback(10)
            asa_data = self._load_data(state)
            self._asa_data = asa_data
            self._aligned_pos = None
            self._input_hash = self._compute_input_hash(state)

            # Stage 2: Preprocess
            log_callback(f"Preprocessing with {state.preprocess_method}...")
            progress_callback(30)

            if state.preprocess_method == "embed_spike_trains":
                from canns.analyzer.data.asa import embed_spike_trains, SpikeEmbeddingConfig

                # Get preprocessing parameters from state or use defaults
                params = state.preprocess_params if state.preprocess_params else {}
                self._embed_hash = self._hash_obj(
                    {
                        "input_hash": self._input_hash,
                        "method": state.preprocess_method,
                        "params": params,
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

                config = SpikeEmbeddingConfig(
                    res=params.get("res", 100),
                    dt=params.get("dt", 0.02),
                    sigma=params.get("sigma", 0.1),
                    smooth=params.get("smooth", True),
                    speed_filter=params.get("speed_filter", True),
                    min_speed=params.get("min_speed", 2.5),
                )

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
                            "params": params,
                        },
                    )
                except Exception as e:
                    log_callback(f"Warning: failed to cache embedding: {e}")
            else:
                # No preprocessing - use spike data directly
                log_callback("No preprocessing - using raw spike data")
                spike = asa_data.get("spike")
                self._embed_hash = self._hash_obj(
                    {
                        "input_hash": self._input_hash,
                        "method": state.preprocess_method,
                        "params": {},
                    }
                )

                # Check if already a dense matrix
                if isinstance(spike, np.ndarray) and spike.ndim == 2:
                    self._embed_data = spike
                    log_callback(f"Using spike matrix shape: {spike.shape}")
                else:
                    log_callback("Warning: spike data is not a dense matrix, some analyses may fail")
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

    async def run_analysis(
        self,
        state: WorkflowState,
        log_callback: Callable[[str], None],
        progress_callback: Callable[[int], None],
    ) -> PipelineResult:
        """Run analysis pipeline based on workflow state.

        Args:
            state: Current workflow state
            log_callback: Callback for log messages
            progress_callback: Callback for progress updates (0-100)

        Returns:
            PipelineResult with success status and artifacts
        """
        t0 = time.time()
        artifacts = {}

        try:
            # Stage 1: Load data
            log_callback("Loading data...")
            progress_callback(10)
            asa_data = self._asa_data if self._asa_data is not None else self._load_data(state)
            if self._input_hash is None:
                self._input_hash = self._compute_input_hash(state)

            # Stage 3: Analysis (mode-dependent)
            log_callback(f"Running {state.analysis_mode} analysis...")
            progress_callback(40)

            mode = state.analysis_mode.lower()
            if mode == "tda":
                artifacts = self._run_tda(asa_data, state, log_callback)
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
                artifacts = self._run_gridscore(asa_data, state, log_callback)
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

    def _load_data(self, state: WorkflowState) -> Dict[str, Any]:
        """Load data based on input mode."""
        if state.input_mode == "asa":
            path = resolve_path(state, state.asa_file)
            data = np.load(path, allow_pickle=True)
            return {k: data[k] for k in data.files}
        elif state.input_mode == "neuron_traj":
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
        else:
            raise ProcessingError(f"Unknown input mode: {state.input_mode}")

    def _run_preprocess(
        self, asa_data: Dict[str, Any], state: WorkflowState
    ) -> Dict[str, Any]:
        """Run preprocessing on ASA data."""
        if state.preprocess_method == "embed_spike_trains":
            from canns.analyzer.data.asa import embed_spike_trains, SpikeEmbeddingConfig

            params = state.preprocess_params
            config = SpikeEmbeddingConfig(
                res=params.get("res", 100),
                dt=params.get("dt", 0.02),
                sigma=params.get("sigma", 0.1),
                smooth=params.get("smooth", True),
                speed_filter=params.get("speed_filter", True),
                min_speed=params.get("min_speed", 2.5),
            )

            spike_main = embed_spike_trains(asa_data, config)
            asa_data["spike_main"] = spike_main

        return asa_data

    def _run_tda(
        self, asa_data: Dict[str, Any], state: WorkflowState, log_callback
    ) -> Dict[str, Path]:
        """Run TDA analysis."""
        from canns.analyzer.data.asa import tda_vis, TDAConfig
        from canns.analyzer.data.asa.tda import _plot_barcode, _plot_barcode_with_shuffle

        # Create output directory
        out_dir = state.workdir / "TDA"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Get parameters
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
        )

        log_callback("Computing persistent homology...")

        if self._embed_data is None:
            raise ProcessingError("No preprocessed data available. Run preprocessing first.")

        persistence_path = out_dir / "persistence.npz"
        barcode_path = out_dir / "barcode.png"

        embed_hash = self._embed_hash or self._hash_obj({"embed": "unknown"})
        tda_hash = self._hash_obj({"embed_hash": embed_hash, "params": params})

        if self._stage_cache_hit(out_dir, tda_hash, [persistence_path, barcode_path]):
            log_callback("♻️ Using cached TDA results.")
            return {"persistence": persistence_path, "barcode": barcode_path}

        result = tda_vis(
            embed_data=self._embed_data,
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
        asa_data: Dict[str, Any],
        persistence_path: Path,
        state: WorkflowState,
        log_callback,
    ) -> Dict[str, Any]:
        """Load cached decoding or run decode_circular_coordinates."""
        from canns.analyzer.data.asa import decode_circular_coordinates

        decode_dir = state.workdir / "CohoMap"
        decode_dir.mkdir(parents=True, exist_ok=True)
        decode_path = decode_dir / "decoding.npz"

        params = state.analysis_params
        decode_params = {
            "real_ground": params.get("real_ground", True),
            "real_of": params.get("real_of", True),
        }
        persistence_hash = self._hash_file(persistence_path)
        decode_hash = self._hash_obj({"persistence_hash": persistence_hash, "params": decode_params})

        meta_path = self._stage_cache_path(decode_dir)
        meta = self._load_cache_meta(meta_path)
        if decode_path.exists() and meta.get("decode_hash") == decode_hash:
            log_callback("♻️ Using cached decoding.")
            return self._load_npz_dict(decode_path)

        log_callback("Decoding circular coordinates...")
        decode_result = decode_circular_coordinates(
            persistence_result=self._load_npz_dict(persistence_path),
            spike_data=asa_data,
            real_ground=decode_params["real_ground"],
            real_of=decode_params["real_of"],
            save_path=str(decode_path),
        )

        meta["decode_hash"] = decode_hash
        meta["persistence_hash"] = persistence_hash
        meta["decode_params"] = decode_params
        self._write_cache_meta(meta_path, meta)
        return decode_result

    def _run_cohomap(
        self, asa_data: Dict[str, Any], state: WorkflowState, log_callback
    ) -> Dict[str, Path]:
        """Run CohoMap analysis (TDA + decode + plotting)."""
        from canns.analyzer.data.asa import plot_cohomap_multi
        from canns.analyzer.visualization import PlotConfigs

        tda_dir = state.workdir / "TDA"
        persistence_path = tda_dir / "persistence.npz"
        if not persistence_path.exists():
            raise ProcessingError("TDA results not found. Run TDA analysis first.")

        out_dir = state.workdir / "CohoMap"
        out_dir.mkdir(parents=True, exist_ok=True)

        decode_result = self._load_or_run_decode(asa_data, persistence_path, state, log_callback)

        cohomap_path = out_dir / "cohomap.png"
        stage_hash = self._hash_obj(
            {
                "decode_hash": self._load_cache_meta(self._stage_cache_path(out_dir)).get("decode_hash"),
                "plot": "cohomap",
            }
        )
        if self._stage_cache_hit(out_dir, stage_hash, [cohomap_path]):
            log_callback("♻️ Using cached CohoMap plot.")
            return {"decoding": out_dir / "decoding.npz", "cohomap": cohomap_path}

        log_callback("Generating cohomology map...")
        pos = self._aligned_pos if self._aligned_pos is not None else asa_data
        config = PlotConfigs.cohomap(show=False, save_path=str(cohomap_path))
        plot_cohomap_multi(
            decoding_result=decode_result,
            position_data={"x": pos["x"], "y": pos["y"]},
            config=config,
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
        self, asa_data: Dict[str, Any], state: WorkflowState, log_callback
    ) -> Dict[str, Path]:
        """Run path comparison visualization."""
        from canns.analyzer.data.asa import (
            align_coords_to_position,
            apply_angle_scale,
            plot_path_compare,
        )
        from canns.analyzer.visualization import PlotConfigs

        tda_dir = state.workdir / "TDA"
        persistence_path = tda_dir / "persistence.npz"
        if not persistence_path.exists():
            raise ProcessingError("TDA results not found. Run TDA analysis first.")

        decode_result = self._load_or_run_decode(asa_data, persistence_path, state, log_callback)

        # Create output directory
        out_dir = state.workdir / "PathCompare"
        out_dir.mkdir(parents=True, exist_ok=True)

        params = state.analysis_params
        angle_scale = params.get("angle_scale", "rad")

        coords = np.asarray(decode_result.get("coords"))
        if coords.ndim != 2:
            raise ProcessingError(f"decode_result['coords'] must be 2D, got {coords.shape}")

        pos = self._aligned_pos if self._aligned_pos is not None else asa_data
        t_full = np.asarray(pos["t"]).ravel()
        x_full = np.asarray(pos["x"]).ravel()
        y_full = np.asarray(pos["y"]).ravel()

        log_callback("Aligning decoded coordinates to position...")
        t_use, x_use, y_use, coords_use, _ = align_coords_to_position(
            t_full=t_full,
            x_full=x_full,
            y_full=y_full,
            coords2=coords,
            use_box=True,
            times_box=decode_result.get("times_box", None),
            interp_to_full=True,
        )
        scale = str(angle_scale) if str(angle_scale) in {"rad", "deg", "unit", "auto"} else "rad"
        coords_use = apply_angle_scale(coords_use, scale)

        out_path = out_dir / "path_compare.png"
        decode_meta = self._load_cache_meta(self._stage_cache_path(state.workdir / "CohoMap"))
        stage_hash = self._hash_obj(
            {
                "persistence": self._hash_file(persistence_path),
                "decode_hash": decode_meta.get("decode_hash"),
                "params": {"angle_scale": scale},
            }
        )
        if self._stage_cache_hit(out_dir, stage_hash, [out_path]):
            log_callback("♻️ Using cached PathCompare plot.")
            return {"path_compare": out_path}

        log_callback("Generating path comparison...")
        config = PlotConfigs.path_compare(show=False, save_path=str(out_path))
        plot_path_compare(x_use, y_use, coords_use, config=config)

        self._write_cache_meta(self._stage_cache_path(out_dir), {"hash": stage_hash})
        return {"path_compare": out_path}

    def _run_cohospace(
        self, asa_data: Dict[str, Any], state: WorkflowState, log_callback
    ) -> Dict[str, Path]:
        """Run cohomology space visualization."""
        from canns.analyzer.data.asa import (
            plot_cohospace_trajectory,
            plot_cohospace_neuron,
            plot_cohospace_population,
        )
        from canns.analyzer.visualization import PlotConfigs

        tda_dir = state.workdir / "TDA"
        persistence_path = tda_dir / "persistence.npz"
        if not persistence_path.exists():
            raise ProcessingError("TDA results not found. Run TDA analysis first.")

        decode_result = self._load_or_run_decode(asa_data, persistence_path, state, log_callback)

        out_dir = state.workdir / "CohoSpace"
        out_dir.mkdir(parents=True, exist_ok=True)

        params = state.analysis_params
        artifacts: Dict[str, Path] = {}

        coords = np.asarray(decode_result.get("coords"))
        coordsbox = np.asarray(decode_result.get("coordsbox"))
        if coords.ndim != 2 or coords.shape[1] < 2:
            raise ProcessingError("decode_result['coords'] must be 2D with >=2 columns.")

        activity = self._embed_data if self._embed_data is not None else np.asarray(asa_data.get("spike"))

        decode_meta = self._load_cache_meta(self._stage_cache_path(state.workdir / "CohoMap"))
        stage_hash = self._hash_obj(
            {
                "persistence": self._hash_file(persistence_path),
                "decode_hash": decode_meta.get("decode_hash"),
                "params": params,
            }
        )
        meta_path = self._stage_cache_path(out_dir)
        required = [out_dir / "cohospace_trajectory.png", out_dir / "cohospace_population.png"]
        neuron_id = params.get("neuron_id")
        if neuron_id is not None:
            required.append(out_dir / f"cohospace_neuron_{neuron_id}.png")

        if self._stage_cache_hit(out_dir, stage_hash, required):
            log_callback("♻️ Using cached CohoSpace plots.")
            artifacts = {
                "trajectory": out_dir / "cohospace_trajectory.png",
            }
            if neuron_id is not None:
                artifacts["neuron"] = out_dir / f"cohospace_neuron_{neuron_id}.png"
            artifacts["population"] = out_dir / "cohospace_population.png"
            return artifacts

        log_callback("Plotting cohomology space trajectory...")
        traj_path = out_dir / "cohospace_trajectory.png"
        traj_cfg = PlotConfigs.cohospace_trajectory(show=False, save_path=str(traj_path))
        plot_cohospace_trajectory(coords=coords[:, :2], times=None, subsample=2, config=traj_cfg)
        artifacts["trajectory"] = traj_path

        neuron_id = params.get("neuron_id", None)
        if neuron_id is not None:
            log_callback(f"Plotting neuron {neuron_id}...")
            neuron_path = out_dir / f"cohospace_neuron_{neuron_id}.png"
            neuron_cfg = PlotConfigs.cohospace_neuron(show=False, save_path=str(neuron_path))
            plot_cohospace_neuron(
                coords=coordsbox[:, :2],
                activity=activity,
                neuron_id=int(neuron_id),
                mode=params.get("mode", "fr"),
                top_percent=float(params.get("top_percent", 5.0)),
                config=neuron_cfg,
            )
            artifacts["neuron"] = neuron_path

        log_callback("Plotting population activity...")
        pop_path = out_dir / "cohospace_population.png"
        pop_cfg = PlotConfigs.cohospace_population(show=False, save_path=str(pop_path))
        neuron_ids = list(range(activity.shape[1]))
        plot_cohospace_population(
            coords=coords[:, :2],
            activity=activity,
            neuron_ids=neuron_ids,
            mode=params.get("mode", "fr"),
            top_percent=float(params.get("top_percent", 5.0)),
            config=pop_cfg,
        )
        artifacts["population"] = pop_path

        self._write_cache_meta(meta_path, {"hash": stage_hash})
        return artifacts

    def _run_fr(
        self, asa_data: Dict[str, Any], state: WorkflowState, log_callback
    ) -> Dict[str, Path]:
        """Run firing rate heatmap analysis."""
        from canns.analyzer.data.asa import compute_fr_heatmap_matrix, save_fr_heatmap_png
        from canns.analyzer.visualization import PlotConfigs

        out_dir = state.workdir / "FR"
        out_dir.mkdir(parents=True, exist_ok=True)

        params = state.analysis_params
        neuron_range = params.get("neuron_range", None)
        time_range = params.get("time_range", None)
        normalize = params.get("normalize", "zscore_per_neuron")

        if self._embed_data is None:
            raise ProcessingError("No preprocessed data available. Run preprocessing first.")

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
            self._embed_data,
            neuron_range=neuron_range,
            time_range=time_range,
            normalize=normalize,
        )

        config = PlotConfigs.fr_heatmap(show=False, save_path=str(out_path))
        save_fr_heatmap_png(fr_matrix, config=config, dpi=200)

        self._write_cache_meta(self._stage_cache_path(out_dir), {"hash": stage_hash})
        return {"fr_heatmap": out_path}

    def _run_frm(
        self, asa_data: Dict[str, Any], state: WorkflowState, log_callback
    ) -> Dict[str, Path]:
        """Run single neuron firing rate map."""
        from canns.analyzer.data.asa import compute_frm, plot_frm
        from canns.analyzer.visualization import PlotConfigs

        out_dir = state.workdir / "FRM"
        out_dir.mkdir(parents=True, exist_ok=True)

        params = state.analysis_params
        neuron_id = int(params.get("neuron_id", 0))
        bins = int(params.get("bin_size", 50))
        smooth_sigma = float(params.get("smooth_sigma", 2.0))

        if self._embed_data is None:
            raise ProcessingError("No preprocessed data available. Run preprocessing first.")

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
            self._embed_data,
            x,
            y,
            neuron_id=neuron_id,
            bins=max(1, bins),
            min_occupancy=1,
            smoothing=True,
            sigma=smooth_sigma,
            nan_for_empty=True,
        )

        config = PlotConfigs.frm(show=False, save_path=str(out_path))
        plot_frm(frm_result.frm, config=config, dpi=200)

        self._write_cache_meta(self._stage_cache_path(out_dir), {"hash": stage_hash})
        return {"frm": out_path}

    def _run_gridscore(
        self, asa_data: Dict[str, Any], state: WorkflowState, log_callback
    ) -> Dict[str, Path]:
        """Run grid score analysis."""
        from canns.analyzer.metrics.spatial_metrics import compute_grid_score

        # Create output directory
        out_dir = state.workdir / "GridScore"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Get parameters
        params = state.analysis_params
        annulus_inner = params.get("annulus_inner", 0.3)
        annulus_outer = params.get("annulus_outer", 0.7)
        bin_size = params.get("bin_size", 2.5)
        smooth_sigma = params.get("smooth_sigma", 2.0)

        # Compute grid scores for all neurons
        log_callback("Computing grid scores...")
        # TODO: Implement batch grid score computation
        # This would require iterating over neurons and computing individual scores

        # Placeholder result
        scores_path = out_dir / "gridscore.npz"
        dist_path = out_dir / "gridscore_distribution.png"

        return {"scores": scores_path, "distribution": dist_path}
