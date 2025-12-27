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

from canns.analyzer.visualization import PlotConfigs
from canns.analyzer.visualization.theta_sweep_plots import (
    create_theta_sweep_grid_cell_animation,
    plot_grid_cell_manifold,
    plot_population_activity_with_theta,
)
from canns.models.basic.theta_sweep_model import (
    DirectionCellNetwork,
    GridCellNetwork,
    calculate_theta_modulation,
)
from canns.task.open_loop_navigation import OpenLoopNavigationTask


def main() -> None:
    # Set up simulation parameters
    np.random.seed(10)
    Env_size = 1.5
    simulate_time = 2.0
    dt = 0.001
    bm.set_dt(dt=1.0)

    # Create and run spatial navigation task
    snt = OpenLoopNavigationTask(
        duration=simulate_time,
        initial_head_direction=11 / 12 * bm.pi,
        width=Env_size,
        height=Env_size,
        start_pos=[Env_size * 15 / 16, Env_size * 1 / 16],
        speed_mean=2.0,
        speed_std=0.0,
        dt=dt,
        speed_coherence_time=10,
        rotational_velocity_std=40 * np.pi / 180,
    )

    snt.get_data()
    snt.calculate_theta_sweep_data()
    snt_data = snt.data

    # Extract trajectory data
    time_steps = snt.run_steps
    position = snt_data.position
    direction = snt_data.hd_angle
    linear_speed_gains = snt_data.linear_speed_gains
    ang_speed_gains = snt_data.ang_speed_gains

    # Show trajectory analysis
    print("Displaying trajectory analysis...")
    snt.show_trajectory_analysis(save_path="open_loop_trajectory_analysis.png", show=False)

    # Create networks
    dc_net = DirectionCellNetwork(
        num=100,
        adaptation_strength=15,
        noise_strength=0.0,
    )

    mapping_ratio = 5
    gc_net = GridCellNetwork(
        num_dc=dc_net.num,
        num_gc_x=100,
        adaptation_strength=8,
        mapping_ratio=mapping_ratio,
        noise_strength=0.0,
    )

    def run_step(i, pos, hd_angle, linear_gain, ang_gain):
        theta_phase, theta_modulation_hd, theta_modulation_gc = calculate_theta_modulation(
            time_step=i,
            linear_gain=linear_gain,
            ang_gain=ang_gain,
            theta_strength_hd=1.0,
            theta_strength_gc=0.5,
            theta_cycle_len=100.0,
            dt=dt,
        )

        dc_net(hd_angle, theta_modulation_hd)
        dc_activity = dc_net.r.value

        gc_net(pos, dc_activity, theta_modulation_gc)
        gc_activity = gc_net.r.value

        return (
            gc_net.center_position.value,
            dc_net.center.value,
            gc_activity,
            gc_net.gc_bump.value,
            dc_activity,
            theta_phase,
            theta_modulation_hd,
            theta_modulation_gc,
        )

    results = bm.for_loop(
        run_step,
        (
            bm.arange(len(position)),
            position,
            direction,
            linear_speed_gains,
            ang_speed_gains,
        )
    )

    (
        internal_position,
        internal_direction,
        gc_netactivity,
        gc_bump,
        dc_netactivity,
        theta_phase,
        theta_modulation_hd,
        theta_modulation_gc,
    ) = results

    print("Plotting population activity with theta modulation...")
    config_pop = PlotConfigs.theta_population_activity_static(
        title="Direction Cell Population Activity with Theta",
        xlabel="Time (s)",
        ylabel="Direction (°)",
        figsize=(10, 4),
        show=True,
    )

    plot_population_activity_with_theta(
        time_steps=time_steps * dt,
        theta_phase=theta_phase,
        net_activity=dc_netactivity,
        direction=direction,
        config=config_pop,
        add_lines=True,
        atol=5e-2,
    )

    print("Plotting grid cell activity on manifold...")
    value_grid_twisted = np.dot(gc_net.coor_transform_inv, gc_net.value_grid.T).T
    grid_cell_activity = gc_netactivity.reshape(-1, gc_net.num_gc_1side, gc_net.num_gc_1side)
    frame_idx = 900

    config_manifold = PlotConfigs.grid_cell_manifold_static(
        title="Grid Cell Activity on Twisted Torus Manifold",
        figsize=(6, 5),
        show=True,
    )

    plot_grid_cell_manifold(
        value_grid_twisted=value_grid_twisted / mapping_ratio,
        grid_cell_activity=grid_cell_activity[frame_idx],
        config=config_manifold,
    )

    print("Creating theta sweep animation...")
    config_animation = PlotConfigs.theta_sweep_animation(
        figsize=(12, 3),
        fps=10,
        save_path="theta_sweep_animation.mp4",
        show=False,
    )

    animation = create_theta_sweep_grid_cell_animation(
        position_data=position,
        direction_data=direction,
        dc_activity_data=dc_netactivity,
        gc_activity_data=gc_netactivity,
        gc_network=gc_net,
        env_size=Env_size,
        mapping_ratio=mapping_ratio,
        dt=dt,
        config=config_animation,
        n_step=10,
        show_progress_bar=True,
        render_backend="auto",
        output_dpi=120,
    )

    if animation is not None:
        print("- Matplotlib animation object created (interactive backend in use).")

    print(f"- Animation saved to: {config_animation.save_path}")


if __name__ == "__main__":
    main()
