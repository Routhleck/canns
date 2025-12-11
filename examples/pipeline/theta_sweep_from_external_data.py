"""Pipeline demo using external trajectory data for theta sweep analysis.

This example demonstrates how to analyze experimental trajectory data using
the ThetaSweepPipeline without needing to understand CANN implementation details.
"""

import numpy as np

from canns.pipeline import ThetaSweepPipeline


def _catmull_rom_chain(control_points: np.ndarray, samples: int) -> np.ndarray:
    """Generate a smooth Catmull–Rom spline through the given control points."""

    if control_points.shape[0] < 4:
        raise ValueError("Catmull–Rom spline requires at least 4 control points.")

    extended = np.vstack([control_points[0], control_points, control_points[-1]])
    segment_count = control_points.shape[0] - 1
    sample_bins = np.linspace(0, samples, num=segment_count + 1, dtype=int)

    curve: list[np.ndarray] = []
    for seg_idx in range(segment_count):
        seg_samples = sample_bins[seg_idx + 1] - sample_bins[seg_idx]
        if seg_samples <= 0:
            continue
        p0, p1, p2, p3 = extended[seg_idx: seg_idx + 4]
        t_values = np.linspace(0.0, 1.0, seg_samples, endpoint=seg_idx == segment_count - 1)
        for t in t_values:
            t2 = t * t
            t3 = t2 * t
            point = 0.5 * (
                (2.0 * p1)
                + (-p0 + p2) * t
                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
            )
            curve.append(point)

    return np.asarray(curve)


def main() -> None:
    # Create example external trajectory shaped as a perturbed multi-node loop
    n_steps = 1000
    dt = 0.002
    env_size = 1.5
    times = np.linspace(0, n_steps * dt, n_steps)

    base_nodes = np.array(
        [
            [0.4, 0.35],
            [0.95, 0.3],
            [1.1, 0.6],
            [0.9, 1.1],
            [0.5, 1.2],
            [0.35, 0.7],
            [0.4, 0.35],
        ]
    )
    rng = np.random.default_rng(1234)
    perturb = rng.normal(scale=0.08, size=base_nodes.shape)
    perturb[[0, -1]] = 0.0
    control_points = np.clip(base_nodes + perturb, 0.05, env_size - 0.05)

    positions = _catmull_rom_chain(control_points, n_steps)

    param = np.linspace(0.0, 1.0, len(positions))
    wobble = 0.025 * np.column_stack(
        (
            np.sin(6 * np.pi * param + np.pi / 6),
            np.sin(3.5 * np.pi * param + np.pi / 3),
        )
    )
    wobble[[0, -1]] = 0.0
    positions = np.clip(positions + wobble, 0.05, env_size - 0.05)

    print("Running theta sweep analysis on external trajectory...")
    print(f"Trajectory: {len(positions)} steps, duration: {times[-1]:.2f}s")

    # Run complete analysis using the pipeline
    pipeline = ThetaSweepPipeline(
        trajectory_data=positions,
        times=times,
        env_size=env_size,
    )
    results = pipeline.run(output_dir="theta_sweep_results")

    print(f"\nAnalysis complete!")
    print(f"Animation saved to: {results['animation_path']}")
    print(f"Plots saved to: theta_sweep_results/")


if __name__ == "__main__":
    main()
