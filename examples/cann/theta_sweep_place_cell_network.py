"""Theta sweep demo with auto-optimized animation backend.

This example keeps the execution wrapped in ``main()`` with an
``if __name__ == "__main__"`` guard so that when the animation code switches to
the imageio backend (which relies on multiprocessing and, on macOS/Windows,
spawns fresh Python processes) the module is not re-imported and executed
multiple times. Removing the guard would cause the entire script to run once per
worker when using the parallel MP4 renderer.
"""

import brainpy.math as bm
import numpy as np

from canns.analyzer.visualization.spike_plots import population_activity_heatmap
from canns.analyzer.visualization.theta_sweep_plots import create_theta_sweep_place_cell_animation
from canns.models.basic.theta_sweep_model import PlaceCellNetwork
from canns.task.open_loop_navigation import TMazeRecessOpenLoopNavigationTask


def main() -> None:
    # Set up simulation parameters
    np.random.seed(10)
    simulate_time = 3.0
    dt = 0.001
    bm.set_dt(dt=1.0)

    # Create and run spatial navigation task with T-maze geometry from Chu et al. 2024
    tmazet = TMazeRecessOpenLoopNavigationTask(
        duration=simulate_time,
        w=0.84,  # Corridor width (m)
        l_s=3.64,  # Stem length (m)
        l_arm=2.36,  # Arm length (m)
        t=1.0,  # T-junction thickness (m)
        start_pos=(0.0, 0.6),
        recess_width=0.2,
        recess_depth=0.2,
        initial_head_direction=1 / 2 * bm.pi,
        speed_mean=1.2,  # Agent speed (m/s)
        speed_std=0.0,
        rotational_velocity_std=0,
        dt=dt,
    )

    tmazet.get_data()
    tmazet.calculate_theta_sweep_data()
    tmazet.set_grid_resolution(0.05, 0.05)
    geodesic_result = tmazet.compute_geodesic_distance_matrix()
    tmazet_data = tmazet.data

    # Extract trajectory data
    time_steps = tmazet.run_steps
    position = tmazet_data.position
    direction = tmazet_data.hd_angle
    linear_speed_gains = tmazet_data.linear_speed_gains
    ang_speed_gains = tmazet_data.ang_speed_gains

    # Show trajectory analysis
    print("Displaying trajectory analysis...")
    tmazet.show_data(show=False, overlay_movement_cost=True, save_path="tmaze_trajectory_analysis.png")
    tmazet.show_geodesic_distance_matrix(show=False, save_path="tmaze_geodesic_distance_matrix.png")

    # Create networks with T-maze parameters from Chu et al. 2024 Table 3
    pc_net = PlaceCellNetwork(
        geodesic_result,
        tau=3.0,  # Fast neural time constant (ms)
        tau_v=150.0,  # Slow adaptation time constant (ms)
        noise_strength=0.05,
        k=1.40,  # Global inhibition strength
        m=1.1,  # Gives effective m ≈ 3.96
        a=0.3,  # Local excitation range
        A=2.3,  # Local excitation strength
        J0=0.25,  # Baseline excitation
        g=20.0,  # Excitatory gain
        conn_noise=0.0,
    )

    # Warmup period: run network for 0.3s at starting position (MATLAB t_start=300ms)
    warmup_time = 0.1  # seconds
    warmup_steps = int(warmup_time / dt)
    print(f"Running network warmup for {warmup_time}s ({warmup_steps} steps)...")

    def warmup_step(i):
        pc_net(position[0], 1.0)  # Run at start position with no theta modulation
        return None

    bm.for_loop(
        warmup_step,
        bm.arange(warmup_steps),
    )
    print("Warmup completed.")

    def run_step(i, pos, vel_gain, theta_strength=0.1, theta_cycle_len=100):
        t = i * bm.get_dt()
        theta_phase = bm.mod(t, theta_cycle_len) / theta_cycle_len
        theta_phase = theta_phase * 2 * bm.pi - bm.pi

        theta_modulation = 1 + theta_strength * vel_gain * bm.cos(theta_phase)

        pc_net(pos, theta_modulation)

        return (
            pc_net.center.value,
            pc_net.r.value,
            theta_phase,
            theta_modulation,
        )

    results = bm.for_loop(
        run_step,
        (
            bm.arange(len(position)),
            position,
            linear_speed_gains,
        )
    )

    (
        internal_position,
        net_activity,
        theta_phase,
        theta_modulation,
    ) = results

    # remove if needed (this animation will need more time)
    create_theta_sweep_place_cell_animation(
        position_data=position,
        pc_activity_data=net_activity,
        pc_network=pc_net,
        navigation_task=tmazet,
        n_step=20,
        fps=10,
        figsize=(14, 5),
        save_path="place_cell_theta_sweep.mp4",
        show=False,  # Don't show to avoid display errors after saving
    )

    # Create population activity heatmap (static visualization)
    print("\nCreating place cell population activity heatmap...")
    population_activity_heatmap(
        activity_data=net_activity,
        dt=dt,
        title="Place Cell Population Activity",
        figsize=(10, 6),
        cmap="viridis",
        save_path="place_cell_population_activity.png",
        show=False,
    )


if __name__ == "__main__":
    main()
