import brainstate as bst
import brainstate.compile
import braintools
import brainunit as u
import jax

from canns.models.basic import CANN1D, CANN1D_SFA

dur1, dur2, dur3 = 100., 2000., 500.


@jax.jit
def get_inp(t):
    pos = u.math.where(t < dur1, 0., u.math.where(t < dur1 + dur2, final_pos * (t - dur1) / (dur2 - dur1), final_pos))
    return cann_sfa.get_stimulus_by_pos(pos)


bst.environ.set(dt=0.1)
cann = CANN1D(num=512)
cann_sfa = CANN1D_SFA(num=512)
cann.init_state()
cann_sfa.init_state()


def run_step(t):
    with bst.environ.context(t=t):
        inp = get_inp(t)
        cann_sfa(inp)
        cann(inp)
        return cann.u.value, cann_sfa.u.value, cann_sfa.v.value, cann_sfa.inp.value


final_pos = cann_sfa.a / cann_sfa.tau_v * 0.6 * dur2

times = u.math.arange(0, dur1 + dur2 + dur3, bst.environ.get_dt())
cann_us, cann_sfa_us, cann_sfa_vs, inps = bst.compile.for_loop(run_step, times, pbar=brainstate.compile.ProgressBar(10))
braintools.visualize.animate_1D(
    dynamical_vars=[{'ys': cann_us, 'xs': cann_sfa.x, 'legend': 'u'},
                    {'ys': cann_sfa_us, 'xs': cann_sfa.x, 'legend': 'u(sfa)'},
                    {'ys': cann_sfa_vs, 'xs': cann_sfa.x, 'legend': 'v(sfa)'},
                    {'ys': inps, 'xs': cann_sfa.x, 'legend': 'Iext'}],
    frame_step=60,
    frame_delay=5,
    save_path='CANN1D_oscillatory_tracking.gif',
)