import brainstate as bst
import brainstate.compile
import brainunit as u
import jax

from canns.analyzer.visualize import energy_landscape_2d_animation
from canns.models.basic import CANN2D
from canns.task.tracking import SmoothTracking2D

bst.environ.set(dt=0.1)

cann = CANN2D(length=100)
cann.init_state()

task_st = SmoothTracking2D(
    cann_instance=cann,
    Iext=([0., 0.], [1., 1.], [0.75, 0.75], [2., 2.], [1.75, 1.75], [3., 3.]),
    duration=(10. ,10., 10., 10., 10.),
    time_step=brainstate.environ.get_dt(),
)
task_st.get_data()

def run_step(t, Iext):
    with bst.environ.context(t=t):
        cann(Iext)
        return cann.u.value, cann.r.value, cann.inp.value

cann_us, cann_rs, inps = bst.compile.for_loop(
    run_step,
    task_st.run_steps,
    task_st.data,
    pbar=brainstate.compile.ProgressBar(10)
)

energy_landscape_2d_animation(
    zs_data=cann_us,
    time_steps_per_second=100,
    fps=20,
    title='CANN2D Encoding',
    xlabel='State X',
    ylabel='State Y',
    clabel='Activity',
    repeat=True,
    save_path='CANN2D_encoding.gif',
    show=False,
)