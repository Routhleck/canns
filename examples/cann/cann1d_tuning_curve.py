import brainpy.math as bm
import brainpy.math as bm
import numpy as np

from canns.analyzer.visualization import PlotConfigs, tuning_curve
from canns.models.basic import CANN1D
from canns.task.tracking import SmoothTracking1D

bm.set_dt(dt=0.1)
cann = CANN1D(num=512, z_min=-np.pi, z_max=np.pi)

task_st = SmoothTracking1D(
    cann_instance=cann,
    Iext=(0., 0., np.pi, 2 * np.pi),
    duration=(2., 20., 20.),
    time_step=bm.get_dt(),
)
task_st.get_data()


def run_step(t, inputs):
    cann(inputs)
    return cann.r.value, cann.inp.value


rs, inps = bm.for_loop(
    run_step,
    (
        task_st.run_steps,
        task_st.data,
    ),
    progress_bar=10
)

# Example of using config-based approach for energy landscape animation
# config_anim = PlotConfigs.energy_landscape_1d_animation(
#     time_steps_per_second=100,
#     fps=20,
#     title='Smooth Tracking 1D',
#     xlabel='State',
#     ylabel='Activity',
#     repeat=True,
#     save_path='smooth_tracking_1d.mp4',
#     show=False
# )
# energy_landscape_1d_animation(
#     data_sets={'u': (cann.x, rs), 'Iext': (cann.x, inps)},
#     config=config_anim
# )

neuron_indices_to_plot = [128, 256, 384]

# Using new config-based approach
config = PlotConfigs.tuning_curve(
    num_bins=50,
    pref_stim=cann.x,
    title='Tuning Curves of Selected Neurons',
    xlabel='Stimulus Position (rad)',
    ylabel='Average Firing Rate',
    show=True,
    save_path=None,
    kwargs={'linewidth': 2, 'marker': 'o', 'markersize': 4}
)

tuning_curve(
    stimulus=task_st.Iext_sequence.squeeze(),
    firing_rates=rs,
    neuron_indices=neuron_indices_to_plot,
    config=config
)

# For comparison, the old-style approach still works:
# tuning_curve(
#     stimulus=task_st.Iext_sequence.squeeze(),
#     firing_rates=rs,
#     neuron_indices=neuron_indices_to_plot,
#     pref_stim=cann.x,
#     num_bins=50,
#     title='Tuning Curves of Selected Neurons (Old Style)',
#     xlabel='Stimulus Position (rad)',
#     ylabel='Average Firing Rate',
#     show=False,  # Set to False to avoid duplicate display
#     save_path=None,
#     linewidth=2,
#     marker='o',
#     markersize=4,
# )
