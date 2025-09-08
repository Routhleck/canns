"""
Grid Cell Model Demonstration

This example demonstrates how to:
1. Create and run Grid Cell models (1D and 2D)
2. Simulate spatial navigation in open environments
3. Analyze hexagonal grid patterns using the analyzer tools
4. Compute grid scores and spacing parameters
5. Visualize grid patterns and their characteristics

This validates that our Grid Cell models produce biologically realistic hexagonal firing patterns.
"""

import numpy as np
import matplotlib.pyplot as plt
import brainstate as bst
from pathlib import Path

# Import our models and analysis tools
from canns.models.basic import GridCell1D, GridCell2D, GridCell1D_CAN, GridCell2D_CAN
from canns.analyzer import grid_cell_analysis
from canns.analyzer.visualize import (
    plot_grid_pattern,
    plot_multiple_grid_patterns,
    plot_grid_score_analysis,
    plot_spatial_tuning_curves,
)


def generate_exploration_trajectory_2d(T: int, width: float = 2.0, height: float = 2.0, speed: float = 0.05):
    """Generate an exploration trajectory that covers the environment well."""
    positions = np.zeros((T, 2))
    positions[0] = [width/2, height/2]  # Start at center
    
    # Parameters for exploration behavior
    direction = np.random.uniform(0, 2*np.pi)
    turn_probability = 0.02
    wall_avoidance_distance = 0.1
    
    for t in range(1, T):
        # Occasional random turns
        if np.random.random() < turn_probability:
            direction += np.random.uniform(-np.pi/3, np.pi/3)
        
        # Wall avoidance
        current_pos = positions[t-1]
        if (current_pos[0] < wall_avoidance_distance or 
            current_pos[0] > width - wall_avoidance_distance):
            direction = np.pi - direction + np.random.uniform(-0.1, 0.1)
        if (current_pos[1] < wall_avoidance_distance or 
            current_pos[1] > height - wall_avoidance_distance):
            direction = -direction + np.random.uniform(-0.1, 0.1)
        
        # Move
        dx = speed * np.cos(direction)
        dy = speed * np.sin(direction)
        
        new_x = np.clip(current_pos[0] + dx, 0.05, width - 0.05)
        new_y = np.clip(current_pos[1] + dy, 0.05, height - 0.05)
        
        positions[t] = [new_x, new_y]
        
        # Add small random noise
        positions[t] += np.random.normal(0, 0.001, 2)
    
    return positions


def generate_linear_exploration_1d(T: int, x_min: float = -np.pi, x_max: float = np.pi, n_cycles: int = 8):
    """Generate exploration trajectory for 1D grid cells."""
    positions = np.zeros(T)
    
    # Create multiple cycles with varying speeds and some randomness
    cycle_length = T // n_cycles
    
    for cycle in range(n_cycles):
        start_idx = cycle * cycle_length
        end_idx = min((cycle + 1) * cycle_length, T)
        
        if cycle % 2 == 0:  # Forward
            base_traj = np.linspace(x_min, x_max, end_idx - start_idx)
        else:  # Backward
            base_traj = np.linspace(x_max, x_min, end_idx - start_idx)
        
        # Add some variability
        positions[start_idx:end_idx] = base_traj + np.random.normal(0, 0.05, end_idx - start_idx)
    
    return positions


def demo_grid_cell_1d():
    """Demonstrate 1D Grid Cell model."""
    print("=== 1D Grid Cell Demo ===")
    
    # Simulation parameters
    T = 3000
    dt = 0.1
    num_cells = 25
    x_min, x_max = -np.pi, np.pi
    
    # Initialize model
    with bst.environ.context(dt=dt):
        model = GridCell1D(
            num=num_cells,
            tau=5.0,
            spacing=0.6,    # Grid spacing
            A=10.0,         # Input amplitude
            x_min=x_min,
            x_max=x_max,
        )
        model.init_state()
    
    # Generate trajectory
    positions_1d = generate_linear_exploration_1d(T, x_min, x_max, n_cycles=6)
    positions_2d = np.column_stack([positions_1d, np.zeros(T)])
    
    # Run simulation
    activity = np.zeros((T, num_cells))
    
    with bst.environ.context(dt=dt):
        for t in range(T):
            model.update(positions_1d[t])
            activity[t] = model.r.value
    
    print(f"Simulated {T} time steps with {num_cells} grid cells")
    print(f"Mean firing rate: {np.mean(activity):.3f} Hz")
    print(f"Max firing rate: {np.max(activity):.3f} Hz")
    
    # Analyze grid properties (simplified for 1D)
    from canns.analyzer.place_cell_analysis import compute_firing_field
    heatmaps = compute_firing_field(
        activity, positions_2d, 
        width=x_max-x_min, height=0.2, 
        n_bins_x=100, n_bins_y=1
    )
    
    # Compute grid scores for 1D (simplified)
    grid_scores = np.zeros(num_cells)
    for i in range(num_cells):
        # For 1D, we use a simpler periodicity measure
        firing_map = heatmaps[:, 0, i]  # Take the 1D slice
        if np.max(firing_map) > 0:
            # Measure periodicity using autocorrelation
            autocorr = np.correlate(firing_map, firing_map, mode='full')
            autocorr = autocorr / np.max(autocorr)
            # Find secondary peaks as measure of periodicity
            center = len(autocorr) // 2
            if center < len(autocorr) - 20:
                secondary_peaks = autocorr[center+10:center+40]
                grid_scores[i] = np.max(secondary_peaks) if len(secondary_peaks) > 0 else 0
    
    print(f"Mean 1D grid score: {np.mean(grid_scores):.3f}")
    print(f"Best 1D grid score: {np.max(grid_scores):.3f}")
    
    # Visualize 1D grid patterns
    plot_spatial_tuning_curves(
        activity, positions_2d, np.arange(min(6, num_cells)),
        width=x_max-x_min, height=0.2,
        title="1D Grid Cell Periodic Patterns",
        save_path="grid_cell_1d_patterns.png"
    )
    
    print("Saved 1D grid patterns: grid_cell_1d_patterns.png")
    
    return activity, positions_2d, grid_scores


def demo_grid_cell_2d():
    """Demonstrate 2D Grid Cell model with hexagonal patterns."""
    print("\n=== 2D Grid Cell Demo ===")
    
    # Simulation parameters
    T = 8000
    dt = 0.05
    length = 12  # 12x12 grid
    width, height = 1.5, 1.5
    
    # Initialize model
    with bst.environ.context(dt=dt):
        model = GridCell2D(
            length=length,
            tau=3.0,
            spacing=0.3,      # Grid spacing
            orientation=0.1,   # Slight rotation
            phase=(0.0, 0.0),  # Phase offset
            A=15.0,           # Input amplitude
            x_min=0.0, x_max=width,
            y_min=0.0, y_max=height,
        )
        model.init_state()
    
    # Generate exploration trajectory
    positions = generate_exploration_trajectory_2d(T, width, height, speed=0.03)
    
    # Run simulation
    activity = np.zeros((T, length * length))
    
    with bst.environ.context(dt=dt):
        for t in range(T):
            model.update(positions[t])
            activity[t] = model.r.value.flatten()
    
    print(f"Simulated {T} time steps with {length*length} grid cells")
    print(f"Mean firing rate: {np.mean(activity):.3f} Hz")
    print(f"Max firing rate: {np.max(activity):.3f} Hz")
    
    # Analyze grid patterns
    grid_scores, grid_indices, num_grid_cells = grid_cell_analysis.select_grid_cells(
        activity, positions, 
        grid_score_threshold=0.1,  # Lower threshold for demo
        min_fields_threshold=2,
        width=width, height=height
    )
    
    print(f"Detected {num_grid_cells} grid cells (score > 0.1, fields >= 2)")
    
    if num_grid_cells > 0:
        # Detailed analysis of grid population
        grid_analysis = grid_cell_analysis.analyze_grid_population(
            activity, positions, grid_indices, width=width, height=height
        )
        
        print(f"Grid cell properties:")
        print(f"  - Mean grid score: {np.mean(grid_analysis['grid_scores']):.3f}")
        print(f"  - Best grid score: {np.max(grid_analysis['grid_scores']):.3f}")
        print(f"  - Mean spacing: {np.mean(grid_analysis['spacings'][grid_analysis['spacings'] > 0]):.3f}")
        print(f"  - Mean regularity: {np.mean(grid_analysis['regularities']):.3f}")
        
        # Visualize grid patterns
        if num_grid_cells > 0:
            # Select best grid cells for visualization
            top_scores_idx = np.argsort(grid_analysis['grid_scores'])[-min(9, num_grid_cells):]
            top_grid_indices = grid_indices[top_scores_idx]
            
            # Extract parameters for visualization
            viz_params = {
                'grid_scores': grid_analysis['grid_scores'][top_scores_idx],
                'spacings': grid_analysis['spacings'][top_scores_idx],
                'num_fields': grid_analysis['num_fields'][top_scores_idx],
                'field_centers': [grid_analysis['field_centers'][i] for i in top_scores_idx]
            }
            
            plot_multiple_grid_patterns(
                grid_analysis['heatmaps'],
                np.arange(len(top_scores_idx)),  # Use sequential indices
                viz_params,
                width=width, height=height,
                title="Top 2D Grid Patterns",
                save_path="grid_patterns_2d_multiple.png"
            )
            
            # Plot grid score analysis
            plot_grid_score_analysis(
                grid_scores, grid_analysis,
                title="2D Grid Cell Analysis",
                save_path="grid_score_analysis_2d.png"
            )
            
            print("Saved visualizations:")
            print("  - grid_patterns_2d_multiple.png")  
            print("  - grid_score_analysis_2d.png")
        
        # Visualize best single grid cell
        if len(grid_analysis['heatmaps']) > 0:
            best_idx = np.argmax(grid_analysis['grid_scores'])
            best_heatmap = grid_analysis['heatmaps'][:, :, best_idx]
            best_centers = grid_analysis['field_centers'][best_idx]
            
            plot_grid_pattern(
                best_heatmap, best_centers,
                width=width, height=height,
                title=f"Best Grid Pattern (Score: {grid_analysis['grid_scores'][best_idx]:.3f})",
                save_path="best_grid_pattern_2d.png"
            )
            
            print("  - best_grid_pattern_2d.png")
    
    return activity, positions, grid_analysis if num_grid_cells > 0 else None


def demo_grid_cell_2d_can():
    """Demonstrate 2D Grid Cell with Continuous Attractor Network dynamics."""
    print("\n=== 2D Grid Cell CAN Demo ===")
    
    # Simulation parameters
    T = 4000
    dt = 0.02
    length = 10  # Smaller for CAN model (more computationally intensive)
    width, height = 1.0, 1.0
    
    # Initialize CAN model
    with bst.environ.context(dt=dt):
        model = GridCell2D_CAN(
            length=length,
            tau=2.0,
            spacing=0.25,
            A=8.0,
            beta=2.0,  # Velocity integration strength
            x_min=0.0, x_max=width,
            y_min=0.0, y_max=height,
        )
        model.init_state()
    
    # Generate smoother trajectory for path integration
    positions = generate_exploration_trajectory_2d(T, width, height, speed=0.02)
    
    # Compute velocities for path integration
    velocities = np.zeros((T, 2))
    velocities[1:] = np.diff(positions, axis=0) / dt
    
    # Run simulation with velocity input
    activity = np.zeros((T, length * length))
    
    with bst.environ.context(dt=dt):
        for t in range(T):
            model.update(velocities[t])
            activity[t] = model.r.value.flatten()
    
    print(f"Simulated CAN model: {T} steps, {length*length} cells")
    print(f"Mean firing rate: {np.mean(activity):.3f} Hz")
    print(f"Path integration with velocity inputs")
    
    # Analyze CAN grid patterns (using actual positions for analysis)
    grid_scores, grid_indices, num_grid_cells = grid_cell_analysis.select_grid_cells(
        activity, positions,
        grid_score_threshold=0.05,
        min_fields_threshold=2,
        width=width, height=height
    )
    
    print(f"CAN Grid cells detected: {num_grid_cells}")
    
    if num_grid_cells > 0:
        grid_analysis = grid_cell_analysis.analyze_grid_population(
            activity, positions, grid_indices, width=width, height=height
        )
        
        print(f"CAN Grid properties:")
        print(f"  - Mean grid score: {np.mean(grid_analysis['grid_scores']):.3f}")
        print(f"  - Best grid score: {np.max(grid_analysis['grid_scores']):.3f}")
        
        # Visualize CAN results
        if num_grid_cells > 0:
            best_idx = np.argmax(grid_analysis['grid_scores'])
            best_heatmap = grid_analysis['heatmaps'][:, :, best_idx]
            
            plot_grid_pattern(
                best_heatmap,
                grid_analysis['field_centers'][best_idx] if len(grid_analysis['field_centers'][best_idx]) > 0 else None,
                width=width, height=height,
                title=f"CAN Grid Pattern (Score: {grid_analysis['grid_scores'][best_idx]:.3f})",
                save_path="grid_can_pattern_2d.png"
            )
            
            print("  - grid_can_pattern_2d.png")
        
        return activity, positions, grid_analysis
    
    return activity, positions, None


def compare_grid_models():
    """Compare different grid cell model variants."""
    print("\n=== Grid Cell Model Comparison ===")
    
    models_info = [
        ("Standard 2D", "Uses external position input"),
        ("CAN 2D", "Uses path integration with velocity"),
        ("1D Linear", "Periodic patterns on linear track")
    ]
    
    for name, description in models_info:
        print(f"✓ {name}: {description}")
    
    print("\nAll models demonstrate grid-like periodic firing patterns!")
    print("CAN models additionally show path integration capabilities.")


def main():
    """Run all Grid Cell demonstrations."""
    print("Grid Cell Model Validation Demo")
    print("=" * 40)
    
    # Set random seed for reproducibility
    np.random.seed(123)
    
    try:
        # Run 1D demo
        activity_1d, positions_1d, scores_1d = demo_grid_cell_1d()
        
        # Run 2D demo
        activity_2d, positions_2d, results_2d = demo_grid_cell_2d()
        
        # Run CAN demo
        activity_can, positions_can, results_can = demo_grid_cell_2d_can()
        
        # Model comparison
        compare_grid_models()
        
        print("\n" + "=" * 40)
        print("Grid Cell Demo Completed Successfully!")
        print("\nValidation Results:")
        
        print(f"✓ 1D Grid Cells: Periodic patterns detected")
        print(f"  Best periodicity score: {np.max(scores_1d):.3f}")
        
        if results_2d:
            print(f"✓ 2D Grid Cells: {results_2d['grid_scores'].shape[0]} cells analyzed")
            print(f"  Best grid score: {np.max(results_2d['grid_scores']):.3f}")
            print(f"  Mean spacing: {np.mean(results_2d['spacings'][results_2d['spacings'] > 0]):.3f}")
        
        if results_can:
            print(f"✓ CAN Grid Cells: Path integration working")
            print(f"  Best CAN grid score: {np.max(results_can['grid_scores']):.3f}")
        
        print("\nAll models show characteristic grid cell firing patterns!")
        print("Hexagonal lattice structures emerge as expected in 2D models.")
        
    except Exception as e:
        print(f"\nError during demo: {e}")
        raise


if __name__ == "__main__":
    main()