import brainstate
import numpy as np

from canns.analyzer.visualize import tuning_curve, energy_landscape_1d_animation
from canns.models.basic import CANN1D, CANN1D_SFA
from canns.task.tracking import SmoothTracking1D

brainstate.environ.set(dt=0.1)
cann = CANN1D(num=512, z_min=-np.pi, z_max=np.pi)
cann.init_state()

task_st = SmoothTracking1D(
    cann_instance=cann,
    Iext=(0., 0., np.pi, 2*np.pi),
    duration=(2., 20., 20.),
    time_step=brainstate.environ.get_dt(),
)
task_st.get_data()

def run_step(t, inputs):
    cann(inputs)
    return cann.r.value, cann.inp.value

rs, inps = brainstate.compile.for_loop(
    run_step,
    task_st.run_steps,
    task_st.data,
    pbar=brainstate.compile.ProgressBar(10)
)

# energy_landscape_1d_animation(
#     {'u': (cann.x, rs), 'Iext': (cann.x, inps)},
#     time_steps_per_second=100,
#     fps=20,
#     title='Smooth Tracking 1D',
#     xlabel='State',
#     ylabel='Activity',
#     repeat=True,
#     save_path='smooth_tracking_1d.gif',
#     show=False,
# )

neuron_indices_to_plot = [128, 256, 384]
tuning_curve(
    stimulus=task_st.Iext_sequence.squeeze(),
    firing_rates=rs,
    neuron_indices=neuron_indices_to_plot,
    pref_stim=cann.x,
    num_bins=50,
    title='Tuning Curves of Selected Neurons',
    xlabel='Stimulus Position (rad)',
    ylabel='Average Firing Rate',
    show=True,
    save_path=None,
    linewidth=2,
    marker='o',
    markersize=4,
)