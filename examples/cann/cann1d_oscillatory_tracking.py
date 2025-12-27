import brainpy.math as bm

from canns.analyzer.visualization import PlotConfigs, energy_landscape_1d_animation
from canns.models.basic import CANN1D
from canns.task.tracking import SmoothTracking1D

bm.set_dt(dt=0.1)
cann = CANN1D(num=512)

task_st = SmoothTracking1D(
    cann_instance=cann,
    Iext=(1., 0.75, 2., 1.75, 3.),
    duration=(10., 10., 10., 10.),
    time_step=bm.get_dt(),
)
task_st.get_data()


def run_step(t, inputs):
    cann(inputs)
    return cann.u.value, cann.inp.value


us, inps = bm.for_loop(
    run_step,
    operands=(task_st.run_steps, task_st.data),
)

# Using new config-based approach
config = PlotConfigs.energy_landscape_1d_animation(
    time_steps_per_second=100,
    fps=20,
    title='Smooth Tracking 1D',
    xlabel='State',
    ylabel='Activity',
    repeat=True,
    save_path='test_smooth_tracking_1d.mp4',
    show=False
)

energy_landscape_1d_animation(
    data_sets={'u': (cann.x, us), 'Iext': (cann.x, inps)},
    config=config
)

# For comparison, the old-style approach still works:
# energy_landscape_1d_animation(
#     {'u': (cann.x, us), 'Iext': (cann.x, inps)},
#     time_steps_per_second=100,
#     fps=20,
#     title='Smooth Tracking 1D (Old Style)',
#     xlabel='State',
#     ylabel='Activity',
#     repeat=True,
#     save_path='test_smooth_tracking_1d_old.mp4',
#     show=False,
# )
