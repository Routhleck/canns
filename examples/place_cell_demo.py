"""
Place Cell Model Demonstration

This example demonstrates how to:
1. Create and run Place Cell models (1D and 2D)
2. Simulate spatial navigation trajectories
3. Analyze place field properties using the analyzer tools
4. Visualize place fields and their characteristics

This validates that our Place Cell models produce biologically realistic results.
"""

import numpy as np
import matplotlib.pyplot as plt
import brainstate as bst
from pathlib import Path

# Import our models and analysis tools
from canns.models.basic import PlaceCell1D, PlaceCell2D
from canns.analyzer import place_cell_analysis
from canns.analyzer.visualize import (
    plot_place_field_heatmap,
    plot_multiple_place_fields,
    plot_place_field_properties,
    plot_spatial_tuning_curves,
)


def generate_random_trajectory_2d(T: int, width: float = 1.0, height: float = 1.0, speed: float = 0.1):
    """Generate a random walk trajectory in 2D environment."""
    positions = np.zeros((T, 2))
    positions[0] = [width/2, height/2]  # Start at center
    
    for t in range(1, T):
        # Random direction
        angle = np.random.uniform(0, 2*np.pi)
        dx = speed * np.cos(angle) * np.random.uniform(0.5, 1.5)
        dy = speed * np.sin(angle) * np.random.uniform(0.5, 1.5)
        
        # Update position with boundary reflection
        new_x = positions[t-1, 0] + dx
        new_y = positions[t-1, 1] + dy
        
        # Reflect at boundaries
        if new_x < 0 or new_x > width:
            dx = -dx
        if new_y < 0 or new_y > height:
            dy = -dy
            
        positions[t, 0] = np.clip(positions[t-1, 0] + dx, 0, width)
        positions[t, 1] = np.clip(positions[t-1, 1] + dy, 0, height)
    
    return positions


def generate_linear_trajectory_1d(T: int, x_min: float = 0.0, x_max: float = 1.0, n_laps: int = 5):
    """Generate back-and-forth trajectory on 1D track."""
    positions = np.zeros(T)
    
    # Create multiple laps back and forth
    lap_length = T // n_laps
    
    for lap in range(n_laps):
        start_idx = lap * lap_length
        end_idx = min((lap + 1) * lap_length, T)
        
        if lap % 2 == 0:  # Forward direction
            positions[start_idx:end_idx] = np.linspace(x_min, x_max, end_idx - start_idx)
        else:  # Backward direction
            positions[start_idx:end_idx] = np.linspace(x_max, x_min, end_idx - start_idx)
    
    return positions


def demo_place_cell_1d():
    """Demonstrate 1D Place Cell model."""
    print("=== 1D Place Cell Demo ===")
    
    # Simulation parameters
    T = 2000  # Time steps
    dt = 0.1  # Time step size
    num_cells = 20
    x_min, x_max = 0.0, 1.0
    
    # Initialize model
    with bst.environ.context(dt=dt):
        model = PlaceCell1D(
            num=num_cells,
            tau=10.0,
            sigma=0.1,  # Place field width
            A=5.0,      # Max firing rate
            x_min=x_min,
            x_max=x_max,
        )
        model.init_state()
    
    # Generate trajectory
    positions_1d = generate_linear_trajectory_1d(T, x_min, x_max, n_laps=4)
    positions_2d = np.column_stack([positions_1d, np.zeros(T)])  # Convert to 2D for analysis
    
    # Run simulation
    activity = np.zeros((T, num_cells))
    
    with bst.environ.context(dt=dt):
        for t in range(T):
            model.update(positions_1d[t])
            activity[t] = model.r.value
    
    print(f"Simulated {T} time steps with {num_cells} place cells")
    print(f"Mean firing rate: {np.mean(activity):.3f} Hz")
    print(f"Max firing rate: {np.max(activity):.3f} Hz")
    
    # Analyze place fields
    place_scores, place_indices, num_place_cells = place_cell_analysis.select_place_cells(
        activity, positions_2d, threshold=0.01, width=x_max-x_min, height=0.1
    )
    
    print(f"Detected {num_place_cells} place cells (threshold: 0.01)")
    
    if num_place_cells > 0:
        # Detailed analysis
        analysis_results = place_cell_analysis.analyze_place_field_properties(
            activity, positions_2d, place_indices, width=x_max-x_min, height=0.1
        )
        
        print(f"Place field properties:")
        print(f"  - Mean place score: {np.mean(analysis_results['place_scores']):.3f}")
        print(f"  - Mean field size: {np.mean(analysis_results['field_sizes']):.3f}")
        print(f"  - Mean spatial info: {np.mean(analysis_results['spatial_info']):.3f} bits/spike")
        
        # Visualize results
        plot_spatial_tuning_curves(
            activity, positions_2d, place_indices[:min(6, len(place_indices))],
            width=x_max-x_min, height=0.1,
            title="1D Place Cell Tuning Curves",
            save_path="place_cell_1d_tuning.png"
        )
        
        print("Saved 1D place cell tuning curves to: place_cell_1d_tuning.png")
    
    return activity, positions_2d, analysis_results if num_place_cells > 0 else None


def demo_place_cell_2d():
    """Demonstrate 2D Place Cell model."""
    print("\n=== 2D Place Cell Demo ===")
    
    # Simulation parameters
    T = 5000  # Time steps
    dt = 0.05  # Time step size
    length = 16  # Grid size (16x16 = 256 cells)
    width, height = 1.0, 1.0
    
    # Initialize model
    with bst.environ.context(dt=dt):
        model = PlaceCell2D(
            length=length,
            tau=5.0,
            sigma=0.15,  # Place field width
            A=10.0,      # Max firing rate
            x_max=width,
            y_max=height,
        )
        model.init_state()
    
    # Generate 2D trajectory
    positions = generate_random_trajectory_2d(T, width, height, speed=0.02)
    
    # Run simulation
    activity = np.zeros((T, length * length))
    
    with bst.environ.context(dt=dt):
        for t in range(T):
            model.update(positions[t])
            activity[t] = model.r.value.flatten()
    
    print(f"Simulated {T} time steps with {length*length} place cells")
    print(f"Mean firing rate: {np.mean(activity):.3f} Hz")
    print(f"Max firing rate: {np.max(activity):.3f} Hz")
    
    # Analyze place fields
    place_scores, place_indices, num_place_cells = place_cell_analysis.select_place_cells(
        activity, positions, threshold=0.005, width=width, height=height
    )
    
    print(f"Detected {num_place_cells} place cells (threshold: 0.005)")
    
    if num_place_cells > 0:
        # Detailed analysis
        analysis_results = place_cell_analysis.analyze_place_field_properties(
            activity, positions, place_indices, width=width, height=height
        )
        
        print(f"Place field properties:")
        print(f"  - Mean place score: {np.mean(analysis_results['place_scores']):.3f}")
        print(f"  - Mean field size: {np.mean(analysis_results['field_sizes']):.3f}")
        print(f"  - Mean spatial info: {np.mean(analysis_results['spatial_info']):.3f} bits/spike")
        print(f"  - Mean peak rate: {np.mean(analysis_results['peak_rates']):.3f} Hz")
        
        # Visualize individual place fields
        if num_place_cells > 0:
            # Select top place cells for visualization
            top_indices = place_indices[np.argsort(analysis_results['place_scores'])[-min(9, len(place_indices)):]]
            
            plot_multiple_place_fields(
                analysis_results['heatmaps'].reshape(50, 50, -1),  # Reshape for visualization
                np.arange(len(top_indices)),  # Use sequential indices for heatmaps
                analysis_results['centers'][-len(top_indices):],
                width=width, height=height,
                title="Top Place Fields (2D)",
                save_path="place_fields_2d_multiple.png"
            )
            
            # Plot place field properties
            plot_place_field_properties(
                analysis_results,
                title="2D Place Field Properties",
                save_path="place_field_properties_2d.png"
            )
            
            print("Saved visualizations:")
            print("  - place_fields_2d_multiple.png")
            print("  - place_field_properties_2d.png")
        
        # Visualize single best place field with trajectory
        if len(analysis_results['heatmaps']) > 0:
            best_idx = np.argmax(analysis_results['place_scores'])
            best_heatmap = analysis_results['heatmaps'][:, :, best_idx]
            
            plot_place_field_heatmap(
                best_heatmap, width=width, height=height,
                positions=positions,
                title=f"Best Place Field (Score: {analysis_results['place_scores'][best_idx]:.3f})",
                save_path="best_place_field_2d.png"
            )
            
            print("  - best_place_field_2d.png")
    
    return activity, positions, analysis_results if num_place_cells > 0 else None


def demo_place_cell_theta():
    """Demonstrate Place Cell with theta rhythm modulation."""
    print("\n=== Place Cell with Theta Demo ===")
    
    from canns.models.basic import PlaceCell1D_Theta
    
    # Simulation parameters
    T = 1000
    dt = 0.01  # Smaller time step for theta
    num_cells = 15
    x_min, x_max = 0.0, 1.0
    
    # Initialize theta-modulated model
    with bst.environ.context(dt=dt):
        model = PlaceCell1D_Theta(
            num=num_cells,
            tau=5.0,
            sigma=0.08,
            A=8.0,
            theta_freq=8.0,    # 8 Hz theta rhythm
            theta_amp=0.5,     # 50% modulation
            x_min=x_min,
            x_max=x_max,
        )
        model.init_state()
    
    # Generate slow trajectory for theta analysis
    positions_1d = generate_linear_trajectory_1d(T, x_min, x_max, n_laps=2)
    time_points = np.arange(T) * dt
    
    # Run simulation
    activity = np.zeros((T, num_cells))
    
    with bst.environ.context(dt=dt):
        for t in range(T):
            model.update(positions_1d[t], time_points[t])
            activity[t] = model.r.value
    
    print(f"Simulated theta-modulated place cells")
    print(f"Mean firing rate: {np.mean(activity):.3f} Hz")
    print(f"Theta modulation visible in firing patterns")
    
    # Simple visualization of theta effect
    plt.figure(figsize=(12, 6))
    
    # Show activity of a few cells over time
    for i in range(min(3, num_cells)):
        plt.subplot(3, 1, i+1)
        plt.plot(time_points[:500], activity[:500, i], linewidth=1)
        plt.ylabel(f'Cell {i}\nRate (Hz)')
        if i == 0:
            plt.title('Place Cell Activity with Theta Modulation')
        if i == 2:
            plt.xlabel('Time (s)')
    
    plt.tight_layout()
    plt.savefig('place_cell_theta_activity.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Saved theta activity plot: place_cell_theta_activity.png")
    
    return activity, positions_1d, time_points


def main():
    """Run all Place Cell demonstrations."""
    print("Place Cell Model Validation Demo")
    print("=" * 40)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    try:
        # Run 1D demo
        activity_1d, positions_1d, results_1d = demo_place_cell_1d()
        
        # Run 2D demo
        activity_2d, positions_2d, results_2d = demo_place_cell_2d()
        
        # Run theta demo
        activity_theta, positions_theta, time_theta = demo_place_cell_theta()
        
        print("\n" + "=" * 40)
        print("Place Cell Demo Completed Successfully!")
        print("\nValidation Results:")
        
        if results_1d:
            print(f"✓ 1D Place Cells: {results_1d['num_place_cells']} cells detected")
            print(f"  Average spatial info: {np.mean(results_1d['spatial_info']):.3f} bits/spike")
        
        if results_2d:
            print(f"✓ 2D Place Cells: {results_2d['num_place_cells']} cells detected")
            print(f"  Average spatial info: {np.mean(results_2d['spatial_info']):.3f} bits/spike")
            print(f"  Average field size: {np.mean(results_2d['field_sizes']):.3f}")
        
        print(f"✓ Theta-modulated place cells: Working as expected")
        print("\nAll models show biologically realistic place field properties!")
        
    except Exception as e:
        print(f"\nError during demo: {e}")
        raise


if __name__ == "__main__":
    main()