import brainstate

from canns.analyzer.visualize import energy_landscape_1d_animation, PlotConfigs, energy_landscape_1d_static
from canns.models.basic import PlaceCell1D
from canns.task.tracking import SmoothTracking1D

brainstate.environ.set(dt=0.1)

# Initialize 1D Place Cell model
place_cell = PlaceCell1D(
    num=20,
    tau=10.0,
    sigma=0.1,
    A=5.0,
    x_min=0.0,
    x_max=1.0,
)
place_cell.init_state()

# Create smooth tracking task
task_st = SmoothTracking1D(
    cann_instance=place_cell,
    Iext=(0.1, 0.3, 0.7, 0.9, 1.5),
    duration=(8., 8., 8., 8.),
    time_step=brainstate.environ.get_dt(),
)
task_st.get_data()

def run_step(t, inputs):
    place_cell(inputs)
    return place_cell.r.value, place_cell.inp.value

rs, inps = brainstate.compile.for_loop(
    run_step,
    task_st.run_steps,
    task_st.data,
    pbar=brainstate.compile.ProgressBar(10)
)

# Create visualization
config = PlotConfigs.energy_landscape_1d_animation(
    time_steps_per_second=20,
    fps=20,
    title='Place Cell 1D Tracking',
    xlabel='Position',
    ylabel='Firing Rate (Hz)',
    repeat=True,
    save_path='place_cell_1d_tracking.gif',
    show=False
)

energy_landscape_1d_animation(
    data_sets={'r': (place_cell.x_centers, rs), 'Input': (place_cell.x_centers, inps)},
    config=config
)

# config = PlotConfigs.energy_landscape_1d_static(
#     title='Place Cell 1D Tracking',
#     xlabel='Position',
#     ylabel='Firing Rate (Hz)',
#     save_path='place_cell_1d_tracking.png',
#     show=True
# )
# index = 100
# energy_landscape_1d_static(
#     data_sets={'r': (place_cell.x_centers, rs[index]), 'Input': (place_cell.x_centers, inps[index])},
#     config=config
# )
