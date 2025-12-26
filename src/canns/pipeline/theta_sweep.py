"""
Theta Sweep Pipeline for External Trajectory Analysis

This module provides a high-level pipeline for experimental scientists to analyze
their trajectory data using CANN theta sweep models without needing to understand
the underlying implementation details.
"""

from pathlib import Path
from typing import Any

import brainpy.math as bm
import numpy as np

from ..analyzer.visualization import PlotConfig
from ..analyzer.visualization.theta_sweep_plots import (
    create_theta_sweep_grid_cell_animation,
    plot_population_activity_with_theta,
)
from ..models.basic.theta_sweep_model import (
    DirectionCellNetwork,
    GridCellNetwork,
    calculate_theta_modulation,
)
from ..task.open_loop_navigation import OpenLoopNavigationTask
from ._base import Pipeline


class ThetaSweepPipeline(Pipeline):
    """
    High-level pipeline for theta sweep analysis of external trajectory data.

    This pipeline abstracts the complex workflow of running CANN theta sweep models
    on experimental trajectory data, making it accessible to researchers who want
    to analyze neural responses without diving into implementation details.

    Example:
        ```python
        # Simple usage - just provide trajectory data
        pipeline = ThetaSweepPipeline(
            trajectory_data=positions,  # shape: (n_steps, 2)
            times=times                 # shape: (n_steps,)
        )

        results = pipeline.run(output_dir="my_results/")
        print(f"Animation saved to: {results['animation_path']}")
        ```
    """

    def __init__(
        self,
        trajectory_data: np.ndarray,
        times: np.ndarray | None = None,
        env_size: float = 2.0,
        dt: float = 0.001,
        direction_cell_params: dict[str, Any] | None = None,
        grid_cell_params: dict[str, Any] | None = None,
        theta_params: dict[str, Any] | None = None,
        spatial_nav_params: dict[str, Any] | None = None,
    ):
        """
        Initialize the theta sweep pipeline.

        Args:
            trajectory_data: Position coordinates with shape (n_steps, 2) for 2D trajectories
            times: Optional time array with shape (n_steps,). If None, uniform time steps will be used
            env_size: Environment size (assumes square environment)
            dt: Simulation time step
            direction_cell_params: Parameters for DirectionCellNetwork. If None, uses defaults
            grid_cell_params: Parameters for GridCellNetwork. If None, uses defaults
            theta_params: Parameters for theta modulation. If None, uses defaults
            spatial_nav_params: Additional parameters for OpenLoopNavigationTask. If None, uses defaults
        """
        super().__init__()
        # Store trajectory data
        self.trajectory_data = np.array(trajectory_data)
        self.times = np.array(times) if times is not None else None
        self.env_size = env_size
        self.dt = dt

        # Validate trajectory data
        self._validate_trajectory_data()

        # Set up default parameters
        self.direction_cell_params = self._get_default_direction_cell_params()
        if direction_cell_params:
            self.direction_cell_params.update(direction_cell_params)

        self.grid_cell_params = self._get_default_grid_cell_params()
        if grid_cell_params:
            self.grid_cell_params.update(grid_cell_params)

        self.theta_params = self._get_default_theta_params()
        if theta_params:
            self.theta_params.update(theta_params)

        self.spatial_nav_params = self._get_default_spatial_nav_params()
        if spatial_nav_params:
            self.spatial_nav_params.update(spatial_nav_params)

        # Initialize components
        self.spatial_nav_task = None
        self.direction_network = None
        self.grid_network = None

    def _validate_trajectory_data(self):
        """
        Validate input trajectory data format and dimensions.

        Checks:
        - Trajectory is 2D array (n_steps, n_dims)
        - Only 2D spatial trajectories (n_dims=2)
        - At least 2 time steps
        - Times array matches trajectory length if provided

        Raises:
            ValueError: If validation fails
        """
        if self.trajectory_data.ndim != 2:
            raise ValueError("trajectory_data must be a 2D array with shape (n_steps, n_dims)")

        n_steps, n_dims = self.trajectory_data.shape
        if n_dims != 2:
            raise ValueError("Currently only 2D trajectories are supported")

        if n_steps < 2:
            raise ValueError("trajectory_data must contain at least 2 time steps")

        if self.times is not None:
            if self.times.shape[0] != n_steps:
                raise ValueError("times array length must match trajectory_data length")

    def _get_default_direction_cell_params(self) -> dict[str, Any]:
        """
        Get default parameters for DirectionCellNetwork initialization.

        Returns:
            dict: Default parameters including:
                - num: 100 neurons
                - adaptation_strength: 15 for SFA dynamics
                - noise_strength: 0.0 (no noise)
        """
        return {
            "num": 100,
            "adaptation_strength": 15,
            "noise_strength": 0.0,
        }

    def _get_default_grid_cell_params(self) -> dict[str, Any]:
        """
        Get default parameters for GridCellNetwork initialization.

        Returns:
            dict: Default parameters including:
                - num_gc_x: 100 neurons per dimension (100x100 grid)
                - adaptation_strength: 8 for SFA dynamics
                - mapping_ratio: 5 (controls grid spacing)
                - noise_strength: 0.0 (no noise)
        """
        return {
            "num_gc_x": 100,
            "adaptation_strength": 8,
            "mapping_ratio": 5,
            "noise_strength": 0.0,
        }

    def _get_default_theta_params(self) -> dict[str, Any]:
        """
        Get default parameters for theta oscillation modulation.

        Returns:
            dict: Default parameters including:
                - theta_strength_hd: 1.0 for direction cells
                - theta_strength_gc: 0.5 for grid cells
                - theta_cycle_len: 100.0 ms per cycle
        """
        return {
            "theta_strength_hd": 1.0,
            "theta_strength_gc": 0.5,
            "theta_cycle_len": 100.0,
        }

    def _get_default_spatial_nav_params(self) -> dict[str, Any]:
        """
        Get default parameters for OpenLoopNavigationTask initialization.

        Returns:
            dict: Default parameters including environment size, dt, etc.
        """
        return {
            "width": self.env_size,
            "height": self.env_size,
            "dt": self.dt,
            "progress_bar": False,
        }

    def _setup_open_loop_navigation_task(self):
        """
        Set up and configure the spatial navigation task with trajectory data.

        Creates OpenLoopNavigationTask, imports external trajectory data,
        and calculates theta sweep parameters (velocity, angular speed, etc.).
        """
        # Calculate duration from trajectory data
        if self.times is not None:
            duration = self.times[-1] - self.times[0]
        else:
            duration = len(self.trajectory_data) * self.dt

        # Create spatial navigation task
        self.spatial_nav_task = OpenLoopNavigationTask(duration=duration, **self.spatial_nav_params)

        # Import external trajectory data
        self.spatial_nav_task.import_data(
            position_data=self.trajectory_data, times=self.times, dt=self.dt
        )

        # Calculate theta sweep data
        self.spatial_nav_task.calculate_theta_sweep_data()

    def _setup_neural_networks(self):
        """
        Initialize and configure direction cell and grid cell networks.

        Creates DirectionCellNetwork and GridCellNetwork instances with
        configured parameters and initializes their states.
        """
        # Create direction cell network
        self.direction_network = DirectionCellNetwork(**self.direction_cell_params)

        # Create grid cell network (ensure consistency with direction network)
        grid_params = self.grid_cell_params.copy()
        grid_params["num_dc"] = self.direction_network.num
        self.grid_network = GridCellNetwork(**grid_params)

    def _run_simulation(self):
        """
        Run the main theta sweep simulation loop.

        Executes time-stepped simulation of direction and grid cell networks
        with theta modulation. Records neural activity, theta phase, and
        decoded positions at each time step.

        Returns:
            dict: Simulation results containing:
                - dc_activity: Direction cell firing rates over time
                - gc_activity: Grid cell firing rates over time
                - gc_center_phase: Grid cell bump centers in phase space
                - gc_center_position: Decoded positions from grid cells
                - theta_phase: Theta oscillation phase over time
        """
        # Set BrainState environment
        bm.set_dt(dt=1.0)

        # Extract data from spatial navigation task
        snt_data = self.spatial_nav_task.data
        position = snt_data.position
        direction = snt_data.hd_angle
        linear_speed_gains = snt_data.linear_speed_gains
        ang_speed_gains = snt_data.ang_speed_gains

        def run_step(i, pos, hd_angle, linear_gain, ang_gain):
            """Single simulation step."""
            theta_phase, theta_modulation_hd, theta_modulation_gc = calculate_theta_modulation(
                time_step=i,
                linear_gain=linear_gain,
                ang_gain=ang_gain,
                theta_strength_hd=self.theta_params["theta_strength_hd"],
                theta_strength_gc=self.theta_params["theta_strength_gc"],
                theta_cycle_len=self.theta_params["theta_cycle_len"],
                dt=self.dt,
            )

            # Update direction cell network
            self.direction_network(hd_angle, theta_modulation_hd)
            dc_activity = self.direction_network.r.value

            # Update grid cell network
            self.grid_network(pos, dc_activity, theta_modulation_gc)
            gc_activity = self.grid_network.r.value

            return (
                self.grid_network.center_position.value,
                self.direction_network.center.value,
                gc_activity,
                self.grid_network.gc_bump.value,
                dc_activity,
                theta_phase,
                theta_modulation_hd,
                theta_modulation_gc,
            )

        # Run compiled simulation loop
        results = bm.for_loop(
            run_step,
            bm.arange(len(position)),
            position,
            direction,
            linear_speed_gains,
            ang_speed_gains,
            pbar=None,
        )

        # Unpack results
        (
            internal_position,
            internal_direction,
            gc_activity,
            gc_bump,
            dc_activity,
            theta_phase,
            theta_modulation_hd,
            theta_modulation_gc,
        ) = results

        # Store simulation results
        self.simulation_results = {
            "internal_position": internal_position,
            "internal_direction": internal_direction,
            "gc_activity": gc_activity,
            "gc_bump": gc_bump,
            "dc_activity": dc_activity,
            "theta_phase": theta_phase,
            "theta_modulation_hd": theta_modulation_hd,
            "theta_modulation_gc": theta_modulation_gc,
            "position": position,
            "direction": direction,
            "linear_speed_gains": linear_speed_gains,
            "ang_speed_gains": ang_speed_gains,
            "time_steps": self.spatial_nav_task.run_steps,
        }

    def run(
        self,
        output_dir: str | Path = "theta_sweep_results",
        save_animation: bool = True,
        save_plots: bool = True,
        show_plots: bool = False,
        animation_fps: int = 10,
        animation_dpi: int = 120,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """
        Run the complete theta sweep pipeline.

        Args:
            output_dir: Directory to save output files
            save_animation: Whether to save the theta sweep animation
            save_plots: Whether to save analysis plots
            show_plots: Whether to display plots interactively
            animation_fps: Frame rate for animation
            animation_dpi: DPI for animation output
            verbose: Whether to print progress messages

        Returns:
            Dictionary containing paths to generated files and analysis data
        """
        self.reset()
        if verbose:
            print("🚀 Starting Theta Sweep Pipeline...")

        # Create output directory
        output_path = self.prepare_output_dir(output_dir)

        # Setup pipeline components
        if verbose:
            print("📊 Setting up spatial navigation task...")
        self._setup_open_loop_navigation_task()

        if verbose:
            print("🧠 Setting up neural networks...")
        self._setup_neural_networks()

        if verbose:
            print("⚡ Running theta sweep simulation...")
        self._run_simulation()

        # Generate outputs
        outputs = {"data": self.simulation_results}

        if save_plots or show_plots:
            outputs.update(self._generate_plots(output_path, show_plots, verbose))

        if save_animation:
            outputs.update(
                self._generate_animation(output_path, animation_fps, animation_dpi, verbose)
            )

        if verbose:
            print("✅ Pipeline completed successfully!")
            print(f"📁 Results saved to: {output_path.absolute()}")

        return self.set_results(outputs)

    def _generate_plots(self, output_path: Path, show_plots: bool, verbose: bool) -> dict[str, str]:
        """
        Generate analysis plots for theta sweep results.

        Creates trajectory analysis and population activity visualizations.

        Args:
            output_path: Directory to save plots
            show_plots: Whether to display plots interactively
            verbose: Whether to print progress messages

        Returns:
            dict: Mapping of plot names to file paths
        """
        plot_outputs = {}

        # Trajectory analysis
        if verbose:
            print("📈 Generating trajectory analysis...")
        trajectory_path = output_path / "trajectory_analysis.png"
        self.spatial_nav_task.show_trajectory_analysis(
            save_path=str(trajectory_path), show=show_plots, smooth_window=50
        )
        plot_outputs["trajectory_analysis"] = str(trajectory_path)

        # Population activity with theta
        if verbose:
            print("📊 Generating population activity plot...")
        config_pop = PlotConfig(
            title="Direction Cell Population Activity with Theta",
            xlabel="Time (s)",
            ylabel="Direction (°)",
            figsize=(10, 4),
            show=show_plots,
            save_path=str(output_path / "population_activity.png"),
        )

        plot_population_activity_with_theta(
            time_steps=self.simulation_results["time_steps"] * self.dt,
            theta_phase=self.simulation_results["theta_phase"],
            net_activity=self.simulation_results["dc_activity"],
            direction=self.simulation_results["direction"],
            config=config_pop,
            add_lines=True,
            atol=5e-2,
        )
        plot_outputs["population_activity"] = str(output_path / "population_activity.png")

        return plot_outputs

    def _generate_animation(
        self, output_path: Path, fps: int, dpi: int, verbose: bool
    ) -> dict[str, str]:
        """
        Generate theta sweep animation showing neural dynamics over time.

        Creates animated visualization of direction and grid cell activity
        with theta phase modulation.

        Args:
            output_path: Directory to save animation
            fps: Frames per second for animation
            dpi: Resolution for animation frames
            verbose: Whether to print progress messages

        Returns:
            dict: Mapping containing 'animation' key with file path
        """
        animation_path = output_path / "theta_sweep_animation.gif"

        config_animation = PlotConfig(
            figsize=(12, 3),
            fps=fps,
            save_path=str(animation_path),
            show=False,
        )

        if verbose:
            print("🎬 Creating theta sweep animation...")
            import sys

            sys.stdout.flush()  # Ensure message is printed before animation starts

        # Brief pause to ensure message ordering
        import time

        time.sleep(0.01)

        create_theta_sweep_grid_cell_animation(
            position_data=self.simulation_results["position"],
            direction_data=self.simulation_results["direction"],
            dc_activity_data=self.simulation_results["dc_activity"],
            gc_activity_data=self.simulation_results["gc_activity"],
            gc_network=self.grid_network,
            env_size=self.env_size,
            mapping_ratio=self.grid_cell_params["mapping_ratio"],
            dt=self.dt,
            config=config_animation,
            n_step=10,
            show_progress_bar=verbose,
            render_backend="auto",
            output_dpi=dpi,
            render_worker_batch_size=2,
        )

        return {"animation_path": str(animation_path)}


# Convenience functions for common use cases


def load_trajectory_from_csv(
    filepath: str | Path,
    x_col: str = "x",
    y_col: str = "y",
    time_col: str | None = "time",
    **kwargs,
) -> dict[str, Any]:
    """
    Load trajectory data from CSV file and run theta sweep analysis.

    Args:
        filepath: Path to CSV file
        x_col: Column name for x coordinates
        y_col: Column name for y coordinates
        time_col: Column name for time data (optional)
        **kwargs: Additional parameters passed to ThetaSweepPipeline

    Returns:
        Dictionary containing analysis results and file paths
    """
    import pandas as pd

    df = pd.read_csv(filepath)

    trajectory_data = df[[x_col, y_col]].values
    times = df[time_col].values if time_col and time_col in df.columns else None

    pipeline = ThetaSweepPipeline(trajectory_data, times, **kwargs)
    return pipeline.run(verbose=True)


def batch_process_trajectories(
    trajectory_list: list, output_base_dir: str = "batch_results", **kwargs
) -> dict[str, dict[str, Any]]:
    """
    Process multiple trajectories in batch.

    Args:
        trajectory_list: List of (trajectory_data, times) tuples or trajectory_data arrays
        output_base_dir: Base directory for batch results
        **kwargs: Additional parameters passed to ThetaSweepPipeline

    Returns:
        Dictionary mapping trajectory indices to results
    """
    batch_results = {}

    for i, trajectory_input in enumerate(trajectory_list):
        print(f"\n🔄 Processing trajectory {i + 1}/{len(trajectory_list)}...")

        if isinstance(trajectory_input, tuple):
            trajectory_data, times = trajectory_input
        else:
            trajectory_data, times = trajectory_input, None

        output_dir = Path(output_base_dir) / f"trajectory_{i:03d}"

        try:
            pipeline = ThetaSweepPipeline(trajectory_data, times, **kwargs)
            results = pipeline.run(output_dir=str(output_dir), verbose=False)
            batch_results[f"trajectory_{i:03d}"] = results
            print(f"✅ Trajectory {i + 1} completed successfully")

        except Exception as e:
            print(f"❌ Error processing trajectory {i + 1}: {e}")
            batch_results[f"trajectory_{i:03d}"] = {"error": str(e)}

    return batch_results
