"""Theta sweep demo with auto-optimized animation backend.

This example keeps the execution wrapped in ``main()`` with an
``if __name__ == "__main__"`` guard so that when the animation code switches to
the imageio backend (which relies on multiprocessing and, on macOS/Windows,
spawns fresh Python processes) the module is not re-imported and executed
multiple times. Removing the guard would cause the entire script to run once per
worker when using the parallel GIF renderer.
"""

import brainstate
import brainunit as u
import numpy as np

from canns.analyzer.plotting.spikes import population_activity_heatmap
from canns.models.basic.theta_sweep_model import PlaceCellNetwork
from canns.task.open_loop_navigation import TMazeRecessOpenLoopNavigationTask


def main() -> None:
    # Set up simulation parameters
    np.random.seed(10)
    simulate_time = 1.0
    dt = 0.001
    brainstate.environ.set(dt=1.0)

    # Create and run spatial navigation task with T-maze geometry from Chu et al. 2024
    tmazet = TMazeRecessOpenLoopNavigationTask(
        duration=simulate_time,
        w=0.84,            # Corridor width (m)
        l_s=3.64,          # Stem length (m)
        l_arm=2.36,        # Arm length (m)
        t=1.0,             # T-junction thickness (m)
        start_pos=(0.0, 0.3),
        recess_width=0.5,
        recess_depth=0.5,
        initial_head_direction=1 / 2 * u.math.pi,
        speed_mean=1.2,    # Agent speed (m/s)
        speed_std=0.0,
        rotational_velocity_std=0,
        dt=dt,
    )

    tmazet.get_data()
    tmazet.calculate_theta_sweep_data()
    tmazet.set_grid_resolution(0.021, 0.021)
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

    # Create networks with T-maze parameters from Chu et al. 2024 Table 3
    # Key: adaptation_strength=190.0 gives m = 190×3/144 ≈ 3.96
    pc_net = PlaceCellNetwork(
        geodesic_result,
        tau=3.0,           # Fast neural time constant (ms)
        tau_v=150.0,       # Slow adaptation time constant (ms)
        noise_strength=0.1,
        k=0.25,            # Global inhibition strength
        adaptation_strength=60.0,  # Gives effective m ≈ 3.96
        a=0.5,             # Local excitation range
        A=0.2,             # Local excitation strength
        J0=0.25,         # Baseline excitation
        g=1.0,            # Excitatory gain
        conn_noise=0.0,
    )
    pc_net.init_state()

    # Warmup period: run network for 0.3s at starting position (MATLAB t_start=300ms)
    warmup_time = 0.1  # seconds
    warmup_steps = int(warmup_time / dt)
    print(f"Running network warmup for {warmup_time}s ({warmup_steps} steps)...")

    def warmup_step(i):
        pc_net(position[0], 1.0)  # Run at start position with no theta modulation
        return None

    brainstate.compile.for_loop(
        warmup_step,
        u.math.arange(warmup_steps),
        pbar=brainstate.compile.ProgressBar(10),
    )
    print("Warmup completed.")

    def run_step(i, pos, vel_gain, theta_strength=0, theta_cycle_len=100):
        t = i * brainstate.environ.get_dt()
        theta_phase = u.math.mod(t, theta_cycle_len) / theta_cycle_len
        theta_phase = theta_phase * 2 * u.math.pi - u.math.pi

        theta_modulation = 1 + theta_strength * vel_gain * u.math.cos(theta_phase)

        pc_net(pos, theta_modulation)

        return (
            pc_net.center.value,
            pc_net.r.value,
            theta_phase,
            theta_modulation,
        )

    results = brainstate.compile.for_loop(
        run_step,
        u.math.arange(len(position)),
        position,
        linear_speed_gains,
        pbar=brainstate.compile.ProgressBar(10),
    )

    (
        internal_position,
        net_activity,
        theta_phase,
        theta_modulation,
    ) = results

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
