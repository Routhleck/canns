import brainstate as bst
import brainstate.compile
import braintools
import brainunit as u
import jax

from canns.models.basic import CANN2D

bst.environ.set(dt=0.1)

cann = CANN2D(length=100)
cann.init_state()

# dur1 = 10.
# dur2 = 20.
# Iext, length = braintools.input.section_input(
#     values=[cann.get_stimulus_by_pos([0., 0.]), 0.],
#     durations=[10., 20.],
#     return_length=True
# )

length = 20
positions = braintools.input.ramp_input(-u.math.pi, u.math.pi, duration=length, t_start=0)
positions = u.math.stack([positions, positions]).T
Iext = jax.vmap(cann.get_stimulus_by_pos)(positions)

def run_step(t, Iext):
    with bst.environ.context(t=t):
        cann(Iext)
        return cann.u.value, cann.r.value, cann.inp.value

times = u.math.arange(0, length, bst.environ.get_dt())
cann_us, cann_rs, inps = bst.compile.for_loop(run_step, times, Iext, pbar=brainstate.compile.ProgressBar(10))
braintools.visualize.animate_2D(
    values=cann_rs.reshape((-1, cann.num)),
    net_size=(cann.length, cann.length),
    dt=bst.environ.get_dt(),
    frame_step=2,
    frame_delay=5,
    save_path='CANN2D_encoding.gif',
)