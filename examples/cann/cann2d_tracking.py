import brainpy.math as bm

from canns.analyzer.visualization import PlotConfigs, energy_landscape_2d_animation
from canns.models.basic import CANN2D
from canns.task.tracking import SmoothTracking2D

bm.set(dt=0.1)

cann = CANN2D(length=100)

task_st = SmoothTracking2D(
    cann_instance=cann,
    Iext=([0., 0.], [1., 1.], [0.75, 0.75], [2., 2.], [1.75, 1.75], [3., 3.]),
    duration=(10., 10., 10., 10., 10.),
    time_step=bm.get_dt(),
)
task_st.get_data()


def run_step(t, Iext):
    cann(Iext)
    return cann.u.value, cann.r.value, cann.inp.value


cann_us, cann_rs, inps = bm.for_loop(
    run_step,
    (
        task_st.run_steps,
        task_st.data,
    ),
)

# Using new config-based approach
config = PlotConfigs.energy_landscape_2d_animation(
    time_steps_per_second=100,
    fps=20,
    title='CANN2D Encoding',
    xlabel='State X',
    ylabel='State Y',
    clabel='Activity',
    repeat=True,
    save_path='CANN2D_encoding.mp4',
    show=False
)

energy_landscape_2d_animation(
    zs_data=cann_us,
    config=config
)

# For comparison, the old-style approach still works:
# energy_landscape_2d_animation(
#     zs_data=cann_us,
#     time_steps_per_second=100,
#     fps=20,
#     title='CANN2D Encoding (Old Style)',
#     xlabel='State X',
#     ylabel='State Y',
#     clabel='Activity',
#     repeat=True,
#     save_path='CANN2D_encoding_old.mp4',
#     show=False,
# )
