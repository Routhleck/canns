"""Visualization functions for fixed point analysis."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from sklearn.decomposition import PCA

from ..visualization import PlotConfig
from .fixed_points import FixedPoints

__all__ = ["plot_fixed_points_2d", "plot_fixed_points_3d"]


def plot_fixed_points_2d(
    fixed_points: FixedPoints,
    state_traj: np.ndarray,
    config: PlotConfig | None = None,
    plot_batch_idx: list[int] | None = None,
    plot_start_time: int = 0,
) -> Figure:
    """Plot fixed points and trajectories in 2D using PCA.

    Args:
        fixed_points: FixedPoints object containing analysis results.
        state_traj: State trajectories [n_batch x n_time x n_states].
        config: Plot configuration. If None, uses default config.
        plot_batch_idx: Batch indices to plot trajectories. If None, plots first 30.
        plot_start_time: Starting time index for trajectory plotting.

    Returns:
        matplotlib Figure object.

    Example:
        >>> from canns.analyzer.slow_points import plot_fixed_points_2d, FixedPoints
        >>> from canns.analyzer.visualization import PlotConfig
        >>> config = PlotConfig(
        ...     title="Fixed Points Analysis",
        ...     figsize=(10, 8),
        ...     save_path="fps_2d.png"
        ... )
        >>> fig = plot_fixed_points_2d(unique_fps, hiddens, config=config)
    """
    # Default config
    if config is None:
        config = PlotConfig(
            title="Fixed Points (2D PCA)",
            xlabel="PC 1",
            ylabel="PC 2",
            figsize=(10, 8),
            show_legend=True,
        )

    n_batch, n_time, n_states = state_traj.shape

    # Default: plot first 30 trajectories
    if plot_batch_idx is None:
        plot_batch_idx = list(range(min(30, n_batch)))

    # Flatten trajectories for PCA
    all_states = []
    for batch_idx in plot_batch_idx:
        all_states.append(state_traj[batch_idx, plot_start_time:, :])
    all_states = np.concatenate(all_states, axis=0)  # [n_samples x n_states]

    # Add fixed points
    if fixed_points.n > 0:
        all_states = np.concatenate([all_states, fixed_points.xstar], axis=0)

    # Perform PCA
    pca = PCA(n_components=2)
    all_pca = pca.fit_transform(all_states)

    # Split back
    n_traj_points = sum(len(state_traj[i, plot_start_time:, :]) for i in plot_batch_idx)
    traj_pca = all_pca[:n_traj_points]
    fps_pca = all_pca[n_traj_points:] if fixed_points.n > 0 else np.array([])

    # Create figure
    fig, ax = plt.subplots(figsize=config.figsize)

    # Plot trajectories
    start_idx = 0
    for batch_idx in plot_batch_idx:
        traj_len = len(state_traj[batch_idx, plot_start_time:, :])
        end_idx = start_idx + traj_len
        ax.scatter(
            traj_pca[start_idx:end_idx, 0],
            traj_pca[start_idx:end_idx, 1],
            c="lightblue",
            s=8.0,
            alpha=1.0,
            label="Trajectories" if batch_idx == plot_batch_idx[0] else "",
            zorder=-1,
        )
        start_idx = end_idx

    # Plot fixed points
    if fixed_points.n > 0:
        # Separate stable and unstable
        stable_mask = fixed_points.is_stable
        unstable_mask = ~stable_mask

        if np.any(unstable_mask):
            ax.scatter(
                fps_pca[unstable_mask, 0],
                fps_pca[unstable_mask, 1],
                c="red",
                marker="x",
                s=200,
                linewidths=3,
                label="Unstable Fixed Points",
                zorder=10,
            )

        if np.any(stable_mask):
            ax.scatter(
                fps_pca[stable_mask, 0],
                fps_pca[stable_mask, 1],
                c="darkred",
                marker="o",
                s=200,
                edgecolors="black",
                linewidths=2,
                label="Stable Fixed Points",
                zorder=10,
            )

    ax.set_xlabel(config.xlabel)
    ax.set_ylabel(config.ylabel)
    ax.set_title(config.title)

    if config.show_legend:
        ax.legend()

    if config.grid:
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save if path specified
    if config.save_path:
        plt.savefig(config.save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved 2D plot to: {config.save_path}")

    if config.show:
        plt.show()

    return fig


def plot_fixed_points_3d(
    fixed_points: FixedPoints,
    state_traj: np.ndarray,
    config: PlotConfig | None = None,
    plot_batch_idx: list[int] | None = None,
    plot_start_time: int = 0,
) -> Figure:
    """Plot fixed points and trajectories in 3D using PCA.

    Args:
        fixed_points: FixedPoints object containing analysis results.
        state_traj: State trajectories [n_batch x n_time x n_states].
        config: Plot configuration. If None, uses default config.
        plot_batch_idx: Batch indices to plot trajectories. If None, plots first 30.
        plot_start_time: Starting time index for trajectory plotting.

    Returns:
        matplotlib Figure object.

    Example:
        >>> from canns.analyzer.slow_points import plot_fixed_points_3d, FixedPoints
        >>> from canns.analyzer.visualization import PlotConfig
        >>> config = PlotConfig(
        ...     title="Fixed Points 3D",
        ...     figsize=(12, 10),
        ...     save_path="fps_3d.png"
        ... )
        >>> fig = plot_fixed_points_3d(unique_fps, hiddens, config=config)
    """

    # Default config
    if config is None:
        config = PlotConfig(
            title="Fixed Points (3D PCA)",
            xlabel="PC 1",
            ylabel="PC 2",
            figsize=(12, 10),
            show_legend=True,
        )

    n_batch, n_time, n_states = state_traj.shape

    # Default: plot first 30 trajectories
    if plot_batch_idx is None:
        plot_batch_idx = list(range(min(30, n_batch)))

    # Flatten trajectories for PCA
    all_states = []
    for batch_idx in plot_batch_idx:
        all_states.append(state_traj[batch_idx, plot_start_time:, :])
    all_states = np.concatenate(all_states, axis=0)  # [n_samples x n_states]

    # Add fixed points
    if fixed_points.n > 0:
        all_states = np.concatenate([all_states, fixed_points.xstar], axis=0)

    # Perform PCA
    pca = PCA(n_components=3)
    all_pca = pca.fit_transform(all_states)

    # Split back
    n_traj_points = sum(len(state_traj[i, plot_start_time:, :]) for i in plot_batch_idx)
    fps_pca = all_pca[n_traj_points:] if fixed_points.n > 0 else np.array([])

    # Compute explained variance
    explained_var = pca.explained_variance_ratio_
    total_var = np.sum(explained_var) * 100

    # Create figure
    fig = plt.figure(figsize=config.figsize)
    ax = fig.add_subplot(111, projection="3d")

    # Plot trajectories as lines
    start_idx = 0
    for i, batch_idx in enumerate(plot_batch_idx):
        traj_len = len(state_traj[batch_idx, plot_start_time:, :])
        end_idx = start_idx + traj_len
        traj_segment = all_pca[start_idx:end_idx]

        ax.plot(
            traj_segment[:, 0],
            traj_segment[:, 1],
            traj_segment[:, 2],
            c="blue",
            alpha=0.3,
            linewidth=0.5,
            label="RNN Trajectories" if i == 0 else "",
        )
        start_idx = end_idx

    # Plot fixed points
    if fixed_points.n > 0:
        # Separate stable and unstable
        stable_mask = fixed_points.is_stable
        unstable_mask = ~stable_mask

        if np.any(unstable_mask):
            ax.scatter(
                fps_pca[unstable_mask, 0],
                fps_pca[unstable_mask, 1],
                fps_pca[unstable_mask, 2],
                c="red",
                marker="x",
                s=200,
                linewidths=3,
                label="Unstable FPs",
                zorder=10,
            )

        if np.any(stable_mask):
            ax.scatter(
                fps_pca[stable_mask, 0],
                fps_pca[stable_mask, 1],
                fps_pca[stable_mask, 2],
                c="darkred",
                marker="o",
                s=200,
                edgecolors="black",
                linewidths=2,
                label="Stable FPs",
                zorder=10,
            )

    # Labels and title
    ax.set_xlabel(config.xlabel if config.xlabel else "PC 1")
    ax.set_ylabel(config.ylabel if config.ylabel else "PC 2")
    ax.set_zlabel("PC 3")

    # Add variance info to title
    title = config.title
    if title:
        title += f"\nVariance: {explained_var[0]:.1%}, {explained_var[1]:.1%}, {explained_var[2]:.1%} (total: {total_var:.1f}%)"
    ax.set_title(title)

    if config.show_legend:
        ax.legend()

    # Set aspect ratio to be equal
    max_range = (
        np.array(
            [
                fps_pca[:, 0].max() - fps_pca[:, 0].min() if fixed_points.n > 0 else 1,
                fps_pca[:, 1].max() - fps_pca[:, 1].min() if fixed_points.n > 0 else 1,
                fps_pca[:, 2].max() - fps_pca[:, 2].min() if fixed_points.n > 0 else 1,
            ]
        ).max()
        / 2.0
    )

    if fixed_points.n > 0:
        mid_x = (fps_pca[:, 0].max() + fps_pca[:, 0].min()) * 0.5
        mid_y = (fps_pca[:, 1].max() + fps_pca[:, 1].min()) * 0.5
        mid_z = (fps_pca[:, 2].max() + fps_pca[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Print PCA info
    print(f"  PCA explained variance: {explained_var}")
    print(f"  Total variance explained: {total_var:.2f}%")

    plt.tight_layout()

    # Save if path specified
    if config.save_path:
        plt.savefig(config.save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved 3D plot to: {config.save_path}")

    if config.show:
        plt.show()

    return fig
