"""Execution helpers for the model gallery TUI."""

from __future__ import annotations

import multiprocessing as mp
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import brainpy.math as bm
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from canns.analyzer.metrics.spatial_metrics import compute_firing_field, gaussian_smooth_heatmaps
from canns.analyzer.visualization import (
    PlotConfigs,
    energy_landscape_1d_static,
    energy_landscape_2d_static,
    plot_firing_field_heatmap,
    tuning_curve,
)
from canns.models.basic import CANN1D, CANN2D, GridCell2DVelocity
from canns.task.open_loop_navigation import OpenLoopNavigationTask
from canns.task.tracking import (
    SmoothTracking1D,
    SmoothTracking2D,
    TemplateMatching1D,
    TemplateMatching2D,
)


@dataclass
class GalleryResult:
    """Result from running a gallery analysis."""

    success: bool
    artifacts: dict[str, Path]
    summary: str
    error: str | None = None
    elapsed_time: float = 0.0


class GalleryRunner:
    """Runner for gallery model analyses."""

    def __init__(self) -> None:
        self._mpl_ready = False

    def _ensure_matplotlib_backend(self) -> None:
        if self._mpl_ready:
            return
        matplotlib.use("Agg", force=True)
        self._mpl_ready = True

    def _ensure_multiprocessing(self) -> None:
        """Stabilize multiprocessing behavior on macOS within threads."""
        if sys.platform == "darwin":
            try:
                mp.set_start_method("fork", force=True)
            except RuntimeError:
                pass

    def _ensure_brainpy_environment(self) -> None:
        """Initialize BrainPy environment for worker threads."""
        try:
            import brainstate.environ as bs_env
            from brainpy.math import defaults as bm_defaults

            bm_defaults.setting()
            bm.set_environment(
                mode=bm.nonbatching_mode,
                bp_object_as_pytree=False,
                numpy_func_return="bp_array",
            )
            bs_env.set(
                mode=bm.nonbatching_mode,
                dt=bm.get_dt(),
                bp_object_as_pytree=False,
                numpy_func_return="bp_array",
            )
        except Exception:
            pass

    async def run(
        self,
        model: str,
        analysis: str,
        model_params: dict[str, Any],
        analysis_params: dict[str, Any],
        output_dir: Path,
        log_callback: Callable[[str], None],
        progress_callback: Callable[[int], None],
    ) -> GalleryResult:
        start_time = time.time()
        artifacts: dict[str, Path] = {}

        try:
            self._ensure_matplotlib_backend()
            self._ensure_multiprocessing()
            self._ensure_brainpy_environment()
            output_dir.mkdir(parents=True, exist_ok=True)

            log_callback(f"Running {model} / {analysis}...")
            progress_callback(5)

            if model == "cann1d":
                output_path = self._run_cann1d(
                    analysis,
                    model_params,
                    analysis_params,
                    output_dir,
                    log_callback,
                    progress_callback,
                )
            elif model == "cann2d":
                output_path = self._run_cann2d(
                    analysis,
                    model_params,
                    analysis_params,
                    output_dir,
                    log_callback,
                    progress_callback,
                )
            elif model == "gridcell":
                output_path = self._run_gridcell(
                    analysis,
                    model_params,
                    analysis_params,
                    output_dir,
                    log_callback,
                    progress_callback,
                )
            else:
                raise ValueError(f"Unknown model: {model}")

            artifacts["output"] = output_path
            elapsed = time.time() - start_time
            progress_callback(100)
            return GalleryResult(
                success=True,
                artifacts=artifacts,
                summary=f"Completed in {elapsed:.1f}s",
                elapsed_time=elapsed,
            )
        except Exception as exc:
            elapsed = time.time() - start_time
            log_callback(f"Error: {exc}")
            log_callback(traceback.format_exc())
            return GalleryResult(
                success=False,
                artifacts=artifacts,
                summary=f"Failed after {elapsed:.1f}s",
                error=str(exc),
                elapsed_time=elapsed,
            )

    def _run_cann1d(
        self,
        analysis: str,
        model_params: dict[str, Any],
        analysis_params: dict[str, Any],
        output_dir: Path,
        log_callback: Callable[[str], None],
        progress_callback: Callable[[int], None],
    ) -> Path:
        seed = model_params["seed"]
        np.random.seed(seed)
        bm.random.seed(seed)
        bm.set_dt(model_params["dt"])

        model = CANN1D(
            num=model_params["num"],
            tau=model_params["tau"],
            k=model_params["k"],
            a=model_params["a"],
            A=model_params["A"],
            J0=model_params["J0"],
        )

        output_path = output_dir / f"cann1d_{analysis}_seed{seed}.png"

        if analysis == "connectivity":
            log_callback("Rendering connectivity matrix...")
            progress_callback(30)
            self._plot_connectivity(model.conn_mat, output_path, title="CANN1D Connectivity")
            return output_path

        if analysis == "energy":
            log_callback("Simulating energy landscape...")
            task = TemplateMatching1D(
                model,
                Iext=analysis_params["energy_pos"],
                duration=analysis_params["energy_duration"],
                time_step=model_params["dt"],
            )
            task.get_data(progress_bar=False)

            def run_step(inputs):
                model(inputs)
                return model.u.value

            us = bm.for_loop(run_step, operands=(task.data,), progress_bar=False)
            select_index = len(task.data) // 2
            config = PlotConfigs.energy_landscape_1d_static(
                title="Energy Landscape 1D",
                xlabel="State",
                ylabel="Activity",
                show=False,
                save_path=str(output_path),
                save_format="png",
            )
            energy_landscape_1d_static(
                data_sets={"u": (np.asarray(model.x), np.asarray(us)[select_index])},
                config=config,
            )
            return output_path

        if analysis == "tuning":
            log_callback("Simulating tuning curves...")
            task = SmoothTracking1D(
                model,
                Iext=(
                    analysis_params["tuning_start"],
                    analysis_params["tuning_mid"],
                    analysis_params["tuning_end"],
                ),
                duration=(analysis_params["tuning_duration"],) * 2,
                time_step=model_params["dt"],
            )
            task.get_data(progress_bar=False)

            def run_step(inputs):
                model(inputs)
                return model.r.value

            rs = bm.for_loop(run_step, operands=(task.data,), progress_bar=False)
            neuron_indices = self._parse_indices(analysis_params["tuning_neurons"], len(model.x))
            config = PlotConfigs.tuning_curve(
                num_bins=analysis_params["tuning_bins"],
                pref_stim=np.asarray(model.x),
                title="Tuning Curves",
                xlabel="Stimulus",
                ylabel="Average Rate",
                show=False,
                save_path=str(output_path),
                save_format="png",
            )
            tuning_curve(
                stimulus=task.Iext_sequence.squeeze(),
                firing_rates=np.asarray(rs),
                neuron_indices=neuron_indices,
                config=config,
            )
            return output_path

        if analysis == "template":
            log_callback("Simulating template matching...")
            task = TemplateMatching1D(
                model,
                Iext=analysis_params["template_pos"],
                duration=analysis_params["template_duration"],
                time_step=model_params["dt"],
            )
            task.get_data(progress_bar=False)

            def run_step(inputs):
                model(inputs)
                return model.u.value, model.inp.value

            us, inps = bm.for_loop(run_step, operands=(task.data,), progress_bar=False)
            select_index = len(task.data) // 2
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(
                np.asarray(model.x), np.asarray(inps)[select_index], "r--", linewidth=2.0, alpha=0.6
            )
            ax.plot(np.asarray(model.x), np.asarray(us)[select_index], "b-", linewidth=2.5)
            ax.grid(True, alpha=0.3)
            ax.set_title("Template Matching", fontsize=12, fontweight="bold")
            fig.tight_layout()
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            return output_path

        if analysis == "manifold":
            log_callback("Computing neural manifold...")
            segment = analysis_params["manifold_segment"]
            warmup = analysis_params["manifold_warmup"]
            iext = (0.0, 0.0, np.pi, 2 * np.pi, -2 * np.pi, 0.0)
            durations = (warmup, segment, segment, segment, segment)
            task = SmoothTracking1D(
                model, Iext=iext, duration=durations, time_step=model_params["dt"]
            )
            task.get_data(progress_bar=False)

            def run_step(t, inputs):
                model(inputs)
                return model.r.value

            rs = bm.for_loop(run_step, (task.run_steps, task.data), progress_bar=False)
            n_warmup = int(warmup / model_params["dt"])
            firing_rates = np.asarray(rs[n_warmup:])
            stimulus_pos = np.asarray(task.Iext_sequence).squeeze()[n_warmup:]
            projected = self._pca_projection(firing_rates, n_components=2)

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(
                projected[:, 0], projected[:, 1], c=stimulus_pos, cmap="viridis", s=2, alpha=0.7
            )
            ax.set_title("Neural Manifold (PC1/PC2)", fontsize=12, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])
            fig.tight_layout()
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            return output_path

        raise ValueError(f"Unsupported analysis for CANN1D: {analysis}")

    def _run_cann2d(
        self,
        analysis: str,
        model_params: dict[str, Any],
        analysis_params: dict[str, Any],
        output_dir: Path,
        log_callback: Callable[[str], None],
        progress_callback: Callable[[int], None],
    ) -> Path:
        seed = model_params["seed"]
        np.random.seed(seed)
        bm.random.seed(seed)
        bm.set_dt(model_params["dt"])

        model = CANN2D(
            length=model_params["length"],
            tau=model_params["tau"],
            k=model_params["k"],
            a=model_params["a"],
            A=model_params["A"],
            J0=model_params["J0"],
        )

        output_path = output_dir / f"cann2d_{analysis}_seed{seed}.png"

        if analysis == "connectivity":
            log_callback("Rendering connectivity matrix...")
            progress_callback(30)
            self._plot_connectivity(model.conn_mat, output_path, title="CANN2D Connectivity")
            return output_path

        if analysis == "energy":
            log_callback("Simulating energy landscape...")
            task = TemplateMatching2D(
                model,
                Iext=(analysis_params["energy_x"], analysis_params["energy_y"]),
                duration=analysis_params["energy_duration"],
                time_step=model_params["dt"],
            )
            task.get_data(progress_bar=False)

            def run_step(inputs):
                model(inputs)
                return model.u.value

            us = bm.for_loop(run_step, operands=(task.data,), progress_bar=False)
            select_index = len(task.data) // 2
            config = PlotConfigs.energy_landscape_2d_static(
                title="Energy Landscape 2D",
                xlabel="State",
                ylabel="Activity",
                show=False,
                save_path=str(output_path),
                save_format="png",
            )
            energy_landscape_2d_static(z_data=np.asarray(us)[select_index], config=config)
            return output_path

        if analysis == "firing_field":
            log_callback("Computing firing field...")
            box_size = analysis_params["field_box"]
            task = OpenLoopNavigationTask(
                duration=analysis_params["field_duration"],
                width=box_size,
                height=box_size,
                start_pos=(box_size / 2.0, box_size / 2.0),
                speed_mean=analysis_params["field_speed"],
                speed_std=analysis_params["field_speed_std"],
                dt=model_params["dt"],
                rng_seed=seed,
                progress_bar=False,
            )
            task.get_data()
            positions = task.data.position

            def run_step(inputs):
                stimulus = model.get_stimulus_by_pos(inputs)
                model(stimulus)
                return model.r.value

            rs = bm.for_loop(run_step, operands=(positions,), progress_bar=False)
            activity = np.asarray(rs).reshape(rs.shape[0], -1)

            firing_fields = compute_firing_field(
                activity,
                np.asarray(positions),
                width=box_size,
                height=box_size,
                M=analysis_params["field_resolution"],
                K=analysis_params["field_resolution"],
            )
            firing_fields = gaussian_smooth_heatmaps(
                firing_fields, sigma=analysis_params["field_sigma"]
            )
            cell_idx = min(64, firing_fields.shape[0] - 1)
            config = PlotConfigs.firing_field_heatmap(
                title=f"Firing Field Cell {cell_idx}",
                show=False,
                save_path=str(output_path),
                save_format="png",
            )
            plot_firing_field_heatmap(firing_fields[cell_idx], config=config)
            return output_path

        if analysis == "trajectory":
            log_callback("Computing trajectory comparison...")
            segment = analysis_params["traj_segment"]
            warmup = analysis_params["traj_warmup"]
            iext = (
                (0.0, 0.0),
                (0.0, 0.0),
                (-2.0, 2.0),
                (2.0, 2.0),
                (2.0, -2.0),
                (-2.0, -2.0),
            )
            durations = (warmup, segment, segment, segment, segment)
            task = SmoothTracking2D(
                model, Iext=iext, duration=durations, time_step=model_params["dt"]
            )
            task.get_data(progress_bar=False)
            true_positions = np.asarray(task.Iext_sequence)

            def run_step(inputs):
                model(inputs)
                return model.r.value

            rs = bm.for_loop(run_step, operands=(task.data,), progress_bar=False)
            decoded = self._decode_cann2d_center(np.asarray(rs), model.length)
            decoded_pos = (decoded / model.length - 0.5) * 2 * np.pi
            warmup_steps = int(warmup / model_params["dt"])
            true_pos = true_positions[warmup_steps:]
            decoded_pos = decoded_pos[warmup_steps:]

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(true_pos[:, 0], true_pos[:, 1], "b-", linewidth=1.5, alpha=0.6)
            ax.plot(decoded_pos[:, 0], decoded_pos[:, 1], "r--", linewidth=1.5, alpha=0.8)
            ax.grid(True, alpha=0.3)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title("Trajectory Comparison", fontsize=12, fontweight="bold")
            fig.tight_layout()
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            return output_path

        if analysis == "manifold":
            log_callback("Computing neural manifold...")
            box_size = 2 * np.pi
            task = OpenLoopNavigationTask(
                duration=analysis_params["manifold_duration"],
                width=box_size,
                height=box_size,
                start_pos=(box_size / 2.0, box_size / 2.0),
                speed_mean=analysis_params["manifold_speed"],
                speed_std=analysis_params["manifold_speed_std"],
                dt=model_params["dt"],
                rng_seed=seed,
                progress_bar=False,
            )
            task.get_data()
            positions = task.data.position

            def run_step(inputs):
                stimulus = model.get_stimulus_by_pos(inputs)
                model(stimulus)
                return model.r.value

            rs = bm.for_loop(run_step, operands=(positions,), progress_bar=False)
            n_warmup = int(analysis_params["manifold_warmup"] / model_params["dt"])
            firing_rates = np.asarray(rs[n_warmup:]).reshape(-1, model.length * model.length)
            stimulus_pos = np.asarray(positions[n_warmup:])

            downsample = max(1, analysis_params["manifold_downsample"])
            firing_rates = firing_rates[::downsample]
            stimulus_pos = stimulus_pos[::downsample]

            projected = self._pca_projection(firing_rates, n_components=2)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(
                projected[:, 0],
                projected[:, 1],
                c=stimulus_pos[:, 0],
                cmap="viridis",
                s=2,
                alpha=0.7,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title("Neural Manifold (PC1/PC2)", fontsize=12, fontweight="bold")
            fig.tight_layout()
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            return output_path

        raise ValueError(f"Unsupported analysis for CANN2D: {analysis}")

    def _run_gridcell(
        self,
        analysis: str,
        model_params: dict[str, Any],
        analysis_params: dict[str, Any],
        output_dir: Path,
        log_callback: Callable[[str], None],
        progress_callback: Callable[[int], None],
    ) -> Path:
        seed = model_params["seed"]
        np.random.seed(seed)
        bm.random.seed(seed)
        bm.set_dt(model_params["dt"])

        output_path = output_dir / f"gridcell_{analysis}_seed{seed}.png"

        if analysis == "connectivity":
            np.random.seed(999)
            bm.random.seed(999)
            model = GridCell2DVelocity(
                length=model_params["length"],
                tau=model_params["tau"],
                alpha=model_params["alpha"],
                W_l=model_params["W_l"],
                lambda_net=model_params["lambda_net"],
            )
            log_callback("Rendering connectivity matrix...")
            progress_callback(30)
            self._plot_connectivity(model.conn_mat, output_path, title="Grid Cell Connectivity")
            return output_path

        model = GridCell2DVelocity(
            length=model_params["length"],
            tau=model_params["tau"],
            alpha=model_params["alpha"],
            W_l=model_params["W_l"],
            lambda_net=model_params["lambda_net"],
        )

        box_size = analysis_params["box_size"]
        start_pos = (box_size / 2.0, box_size / 2.0)

        if analysis == "energy":
            log_callback("Computing energy landscape...")
            task = OpenLoopNavigationTask(
                duration=analysis_params["energy_duration"],
                width=box_size,
                height=box_size,
                start_pos=start_pos,
                speed_mean=analysis_params["energy_speed"],
                speed_std=analysis_params["energy_speed_std"],
                dt=bm.get_dt(),
                rng_seed=seed,
                progress_bar=False,
            )
            task.get_data()
            model.heal_network(
                num_healing_steps=analysis_params["energy_heal_steps"],
                dt_healing=1e-4,
            )

            def run_step(vel):
                model(vel)
                return model.s.value

            us = bm.for_loop(run_step, operands=(task.data.velocity,), progress_bar=False)
            select_index = int(task.total_steps * 0.75)
            energy_data = np.asarray(us)[select_index].reshape(model.length, model.length)
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(energy_data, cmap="viridis", origin="lower")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title("Energy Landscape", fontsize=12, fontweight="bold")
            fig.tight_layout()
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            return output_path

        if analysis == "firing_field":
            log_callback("Computing firing field...")
            from canns.analyzer.metrics.systematic_ratemap import compute_systematic_ratemap

            ratemaps = compute_systematic_ratemap(
                model,
                box_width=box_size,
                box_height=box_size,
                resolution=analysis_params["field_resolution"],
                speed=analysis_params["field_speed"],
                num_batches=analysis_params["field_batches"],
                verbose=False,
            )
            firing_fields = np.transpose(ratemaps, (2, 0, 1))
            firing_fields = gaussian_smooth_heatmaps(
                firing_fields, sigma=analysis_params["field_sigma"]
            )
            cell_idx = model.num // 2
            config = PlotConfigs.firing_field_heatmap(
                title=f"Grid Cell Field {cell_idx}",
                show=False,
                save_path=str(output_path),
                save_format="png",
            )
            plot_firing_field_heatmap(firing_fields[cell_idx], config=config)
            return output_path

        if analysis == "path_integration":
            log_callback("Computing path integration...")
            model.heal_network(
                num_healing_steps=analysis_params["path_heal_steps"],
                dt_healing=1e-4,
            )
            task = OpenLoopNavigationTask(
                duration=analysis_params["path_duration"],
                width=box_size,
                height=box_size,
                start_pos=start_pos,
                speed_mean=analysis_params["path_speed"],
                speed_std=analysis_params["path_speed_std"],
                dt=analysis_params["path_dt"],
                rng_seed=seed,
                progress_bar=False,
            )
            task.get_data()
            true_positions = np.asarray(task.data.position)

            def run_step(vel):
                model(vel)
                return model.r.value

            activities = bm.for_loop(run_step, operands=(task.data.velocity,), progress_bar=False)
            activities = np.asarray(activities)
            blob_centers = GridCell2DVelocity.track_blob_centers(activities, model.length)
            blob_displacement = np.diff(blob_centers, axis=0)
            displacement_norm = np.linalg.norm(blob_displacement, axis=1)
            jump_indices = np.where(displacement_norm > 3.0)[0]
            for idx in jump_indices:
                if 0 < idx < len(blob_displacement) - 1:
                    blob_displacement[idx] = (
                        blob_displacement[idx - 1] + blob_displacement[idx + 1]
                    ) / 2
            estimated_pos_neuron = np.cumsum(blob_displacement, axis=0)

            true_pos_rel = true_positions - true_positions[0]
            true_pos_aligned = true_pos_rel[: len(estimated_pos_neuron)]
            X = estimated_pos_neuron.reshape(-1)
            y = true_pos_aligned.reshape(-1)
            scale = np.dot(X, y) / (np.dot(X, X) + 1e-8)
            estimated_pos = scale * estimated_pos_neuron + true_positions[0]

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(
                true_positions[: len(estimated_pos), 0],
                true_positions[: len(estimated_pos), 1],
                "b-",
                alpha=0.5,
                linewidth=1.5,
                label="True",
            )
            ax.plot(
                estimated_pos[:, 0],
                estimated_pos[:, 1],
                "r-",
                alpha=0.7,
                linewidth=1.5,
                label="Estimated",
            )
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.3)
            ax.set_title("Path Integration", fontsize=12, fontweight="bold")
            fig.tight_layout()
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            return output_path

        if analysis == "manifold":
            log_callback("Computing neural manifold...")
            from canns.analyzer.metrics.systematic_ratemap import compute_systematic_ratemap

            ratemaps = compute_systematic_ratemap(
                model,
                box_width=box_size,
                box_height=box_size,
                resolution=analysis_params["field_resolution"],
                speed=analysis_params["field_speed"],
                num_batches=analysis_params["field_batches"],
                verbose=False,
            )
            firing_fields = np.transpose(ratemaps, (2, 0, 1))
            firing_fields = gaussian_smooth_heatmaps(
                firing_fields, sigma=analysis_params["field_sigma"]
            )
            data_for_pca = firing_fields.reshape(firing_fields.shape[0], -1).T
            projected = self._pca_projection(data_for_pca, n_components=3)
            fig = plt.figure(figsize=(6, 5))
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(
                projected[:, 0],
                projected[:, 1],
                projected[:, 2],
                c=projected[:, 2],
                cmap="viridis",
                s=1,
                alpha=0.7,
            )
            ax.axis("off")
            ax.set_title("Grid Cell Manifold", fontsize=12, fontweight="bold")
            fig.tight_layout()
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            return output_path

        raise ValueError(f"Unsupported analysis for Grid Cell: {analysis}")

    def _plot_connectivity(self, conn_mat: Any, output_path: Path, title: str) -> None:
        data = np.asarray(conn_mat)
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(data, cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("Neuron Index")
        ax.set_ylabel("Neuron Index")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _pca_projection(self, data: np.ndarray, n_components: int = 3) -> np.ndarray:
        centered = data - np.mean(data, axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        components = vt[:n_components].T
        return centered @ components

    def _decode_cann2d_center(self, activities: np.ndarray, length: int) -> np.ndarray:
        from scipy.ndimage import center_of_mass, gaussian_filter, label

        T = len(activities)
        n = length
        activities_2d = activities.reshape(T, n, n)
        smoothed = np.array([gaussian_filter(activities_2d[t], sigma=1) for t in range(T)])
        thresholds = smoothed.mean(axis=(1, 2)) + 0.5 * smoothed.std(axis=(1, 2))
        binary_images = smoothed > thresholds[:, None, None]

        centers = []
        for i in range(T):
            labeled, num_features = label(binary_images[i])
            if num_features > 0:
                blob_centers = np.array(
                    center_of_mass(binary_images[i], labeled, range(1, num_features + 1))
                )
                if blob_centers.ndim == 1:
                    blob_centers = blob_centers.reshape(1, -1)
                blob_centers = blob_centers[:, [1, 0]]
                dist = np.linalg.norm(blob_centers - n / 2, axis=1)
                best_center = blob_centers[np.argmin(dist)]
            else:
                best_center = centers[-1] if centers else np.array([n / 2, n / 2])
            centers.append(best_center)
        return np.array(centers)

    def _parse_indices(self, raw: str, max_size: int) -> list[int]:
        cleaned = [p.strip() for p in raw.split(",") if p.strip()]
        indices: list[int] = []
        for part in cleaned:
            try:
                idx = int(part)
            except ValueError:
                continue
            if 0 <= idx < max_size:
                indices.append(idx)
        if not indices:
            indices = [max_size // 4, max_size // 2, (3 * max_size) // 4]
        return indices
