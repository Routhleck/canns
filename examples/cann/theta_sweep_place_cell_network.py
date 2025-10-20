"""Theta sweep demo with auto-optimized animation backend.

This example keeps the execution wrapped in ``main()`` with an
``if __name__ == "__main__"`` guard so that when the animation code switches to
the imageio backend (which relies on multiprocessing and, on macOS/Windows,
spawns fresh Python processes) the module is not re-imported and executed
multiple times. Removing the guard would cause the entire script to run once per
worker when using the parallel GIF renderer.
"""

import numpy as np
import brainstate
import brainunit as u

from canns.analyzer.theta_sweep import create_theta_sweep_place_cell_animation
from canns.models.basic.theta_sweep_model import PlaceCellNetwork
from canns.task.open_loop_navigation import TMazeOpenLoopNavigationTask


def main() -> None:
    # Set up simulation parameters
    np.random.seed(10)
    simulate_time = 2.0
    dt = 0.001
    brainstate.environ.set(dt=1.0)

    # Create and run spatial navigation task
    tmazet = TMazeOpenLoopNavigationTask(
        duration=simulate_time,
        start_pos=(0.0, 0.15),
        initial_head_direction=1/2 * u.math.pi,
        speed_mean=0.5,
        speed_std=0.0,
        rotational_velocity_std=0,
        dt=dt,
    )

    tmazet.get_data()
    tmazet.calculate_theta_sweep_data()
    tmazet.set_grid_resolution(0.025, 0.025)
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
    tmazet.show_data(show=True, overlay_movement_cost=True)

    # Create networks
    pc_net = PlaceCellNetwork(
        geodesic_result
    )
    pc_net.init_state()

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

    # Create animation visualization
    print("\nCreating place cell animation...")
    create_theta_sweep_place_cell_animation(
        position_data=position,
        pc_activity_data=net_activity,
        pc_network=pc_net,
        navigation_task=tmazet,
        dt=dt,
        n_step=10,
        fps=10,
        figsize=(14, 5),
        save_path="place_cell_theta_sweep.gif",
        show=False,  # Don't show to avoid display errors after saving
    )



if __name__ == "__main__":
    main()
