import brainstate as bst
import brainstate.compile
import brainunit as u

from canns.analyzer.visualize import energy_landscape_2d_animation, PlotConfigs
from canns.models.basic import PlaceCell2D
from canns.task.tracking import SmoothTracking2D

bst.environ.set(dt=0.1)

# Initialize 2D Place Cell model
place_cell = PlaceCell2D(
    length=20,
    tau=5.0,
    sigma=0.15,
    A=10.0,
    x_max=1.0,
    y_max=1.0,
)
place_cell.init_state()

# Create smooth tracking task
task_st = SmoothTracking2D(
    cann_instance=place_cell,
    Iext=([0.2, 0.2], [0.8, 0.3], [0.7, 0.8], [0.3, 0.7], [0.5, 0.5]),
    duration=(10., 10., 10., 10.),
    time_step=brainstate.environ.get_dt(),
)
task_st.get_data()

def run_step(t, Iext):
    with bst.environ.context(t=t):
        place_cell(Iext)
        return place_cell.r.value, place_cell.inp.value

place_rs, inps = bst.compile.for_loop(
    run_step,
    task_st.run_steps,
    task_st.data,
    pbar=brainstate.compile.ProgressBar(10)
)

# Create visualization
config = PlotConfigs.energy_landscape_2d_animation(
    time_steps_per_second=100,
    fps=20,
    title='Place Cell 2D Tracking',
    xlabel='X Position',
    ylabel='Y Position', 
    clabel='Firing Rate (Hz)',
    repeat=True,
    save_path='place_cell_2d_tracking.gif',
    show=False
)

energy_landscape_2d_animation(
    zs_data=place_rs,
    config=config
)