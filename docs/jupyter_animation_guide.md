# Jupyter Notebook Animation Support

As of this update, CANNS automatically detects when animations are being created in Jupyter notebooks and displays them as interactive HTML/JavaScript visualizations instead of trying to open external windows.

## Features

- **Automatic Detection**: No code changes needed - animations automatically adapt to the environment
- **Interactive Display**: Animations render as HTML/JavaScript with play/pause controls in Jupyter
- **Backward Compatible**: Works seamlessly with existing code and non-Jupyter environments
- **No Extra Dependencies**: Uses matplotlib's built-in `to_jshtml()` method

## Usage

Simply use the animation functions as you normally would with `show=True`:

```python
from canns.analyzer.plotting import PlotConfigs, energy_landscape_1d_animation

config = PlotConfigs.energy_landscape_1d_animation(
    time_steps_per_second=100,
    fps=20,
    title='Energy Landscape',
    show=True  # Automatically adapts to Jupyter or non-Jupyter environment
)

ani = energy_landscape_1d_animation(
    data_sets={'u': (x_data, y_data)},
    config=config
)
```

## How It Works

When you call an animation function with `show=True`:

1. **In Jupyter Notebook**: The animation is automatically rendered as interactive HTML/JavaScript inline
2. **Outside Jupyter**: The animation opens in a matplotlib window as before

The detection is handled by the `is_jupyter_environment()` function which checks for:
- IPython availability
- Running in a ZMQInteractiveShell (Jupyter notebook) vs TerminalInteractiveShell (IPython terminal)

## Supported Animation Functions

All major animation functions now support automatic Jupyter detection:

- `energy_landscape_1d_animation()`
- `energy_landscape_2d_animation()`
- `create_theta_sweep_place_cell_animation()`
- `create_theta_sweep_grid_cell_animation()`

## Manual Control (Advanced)

If you need explicit control over the display method, you can use the utilities directly:

```python
from canns.analyzer.plotting.jupyter_utils import (
    is_jupyter_environment, 
    display_animation_in_jupyter
)

# Check if running in Jupyter
if is_jupyter_environment():
    print("Running in Jupyter notebook")
    
# Manually display animation in Jupyter
if is_jupyter_environment():
    display_animation_in_jupyter(animation, format='jshtml')
```

### Display Formats

The `display_animation_in_jupyter()` function supports two formats:

- `'jshtml'` (default): JavaScript-based animation, no external dependencies needed
- `'html5'`: HTML5 video tag (requires ffmpeg or similar video encoder)

## Saving Animations

You can still save animations to files while displaying them in Jupyter:

```python
config = PlotConfigs.energy_landscape_1d_animation(
    save_path='animation.gif',  # Save to file
    show=True                    # Also display in Jupyter
)
```

## Troubleshooting

### Animation not displaying in Jupyter

If animations don't display properly:

1. Ensure you're running in a Jupyter notebook (not JupyterLab console or IPython terminal)
2. Check that IPython is installed (should be included with jupyter/notebook)
3. Try restarting the notebook kernel

### "Figure" objects appearing instead of animations

This usually means:
- `show=False` was set (intended behavior)
- The figure was closed before rendering (check for manual `plt.close()` calls)

## Example Notebook

See `/tmp/jupyter_animation_demo.ipynb` for a complete working example demonstrating:
- Jupyter environment detection
- Animation display
- Saving animations while displaying

## Implementation Details

The implementation is located in:
- `src/canns/analyzer/plotting/jupyter_utils.py`: Core utilities
- `src/canns/analyzer/plotting/energy.py`: Energy landscape animations
- `src/canns/analyzer/theta_sweep.py`: Theta sweep animations

Tests are in:
- `tests/analyzer/test_jupyter_integration.py`: Jupyter detection tests
