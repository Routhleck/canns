import brainstate as bst
import braintools
import brainunit as u
import jax

from canns.models import CANN1D

dur1, dur2, dur3 = 100., 2000., 500.


@jax.jit
def get_inp(t):
    pos = u.math.where(t < dur1, 0., u.math.where(t < dur1 + dur2, final_pos * (t - dur1) / (dur2 - dur1), final_pos))
    inp = cann.get_stimulus_by_pos(pos)
    return inp


bst.environ.set(dt=0.1)
cann = CANN1D(num=512)
cann.init_state()


def run_step(t):
    with bst.environ.context(t=t):
        cann(get_inp(t))
        return cann.u.value, cann.v.value, cann.inp.value


final_pos = cann.a / cann.tau_v * 0.6 * dur2

times = u.math.arange(0, dur1 + dur2 + dur3, bst.environ.get_dt())
us, vs, inps = bst.compile.for_loop(run_step, times)
braintools.visualize.animate_1D(
    dynamical_vars=[{'ys': us, 'xs': cann.x, 'legend': 'u'},
                    {'ys': vs, 'xs': cann.x, 'legend': 'v'},
                    {'ys': inps, 'xs': cann.x, 'legend': 'Iext'}],
    frame_step=10,
    frame_delay=5,
    save_path='cann1d_oscillatory_tracking.gif',
)