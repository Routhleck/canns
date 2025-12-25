import os

import brainpy as bp
import brainpy.math as bm
import numpy as np

from canns.analyzer.visualization import (
    PlotConfig,
    PlotConfigs,
    average_firing_rate_plot,
    energy_landscape_1d_animation,
    energy_landscape_1d_static,
    energy_landscape_2d_animation,
    energy_landscape_2d_static,
    raster_plot,
    tuning_curve,
)
from canns.analyzer.metrics.utils import firing_rate_to_spike_train, normalize_firing_rates
from canns.task.tracking import PopulationCoding1D, PopulationCoding2D, SmoothTracking1D
from canns.models.basic import CANN1D, CANN2D


def test_energy_landscape_1d():
    bm.set_dt(dt=0.1)
    cann = CANN1D(num=32)

    task_pc = PopulationCoding1D(
        cann_instance=cann,
        before_duration=10.,
        after_duration=10.,
        duration=20.,
        Iext=0.,
        time_step=bm.get_dt(),
    )
    task_pc.get_data()

    def run_step(t, inputs):
        cann(inputs)
        return cann.u.value, cann.inp.value

    us, inps = bm.for_loop(
        run_step,
        (
            task_pc.run_steps,
            task_pc.data
        )
    )

    # Test with new config-based approach
    output_path_static = 'test_energy_landscape_1d_static.png'
    config = PlotConfigs.energy_landscape_1d_static(
        title='Population Coding 1D (Static)',
        xlabel='State',
        ylabel='Activity',
        save_path=output_path_static,
        show=False
    )
    energy_landscape_1d_static(
        data_sets={'u': (cann.x, us[int(task_pc.total_steps/2)]), 'Iext': (cann.x, inps[int(task_pc.total_steps/2)])},
        config=config
    )
    assert os.path.isfile(output_path_static), f"Output file {output_path_static} was not created."
    
    # Test backward compatibility
    output_path_static_old = 'test_energy_landscape_1d_static_old.png'
    energy_landscape_1d_static(
        {'u': (cann.x, us[int(task_pc.total_steps/2)]), 'Iext': (cann.x, inps[int(task_pc.total_steps/2)])},
        title='Population Coding 1D (Static - Old Style)',
        xlabel='State',
        ylabel='Activity',
        save_path=output_path_static_old,
        show=False,
    )
    assert os.path.isfile(output_path_static_old), f"Output file {output_path_static_old} was not created."

    # Test with new config-based approach
    output_path_animation = 'test_energy_landscape_1d_animation.gif'
    config_anim = PlotConfigs.energy_landscape_1d_animation(
        time_steps_per_second=100,
        fps=20,
        title='Population Coding 1D (Animation)',
        xlabel='State',
        ylabel='Activity',
        repeat=True,
        save_path=output_path_animation,
        show=False
    )
    energy_landscape_1d_animation(
        data_sets={'u': (cann.x, us), 'Iext': (cann.x, inps)},
        config=config_anim
    )
    assert os.path.isfile(output_path_animation), f"Output file {output_path_animation} was not created."

def test_energy_landscape_2d():
    bm.set_dt(dt=0.1)
    cann = CANN2D(length=4)

    task_pc = PopulationCoding2D(
        cann_instance=cann,
        before_duration=10.,
        after_duration=10.,
        duration=20.,
        Iext=[0., 0.],
        time_step=bm.get_dt(),
    )
    task_pc.get_data()

    def run_step(t, inputs):
        cann(inputs)
        return cann.u.value, cann.r.value, cann.inp.value

    us, rs, inps = bm.for_loop(
        run_step,
        (
            task_pc.run_steps,
            task_pc.data
        )
    )

    # Test with new config-based approach
    output_path_static = 'test_energy_landscape_2d_static.png'
    config = PlotConfigs.energy_landscape_2d_static(
        title='Population Coding 2D (Static)',
        xlabel='State X',
        ylabel='State Y',
        clabel='Activity',
        save_path=output_path_static,
        show=False
    )
    energy_landscape_2d_static(
        z_data=us[int(task_pc.total_steps/2)],
        config=config
    )
    assert os.path.isfile(output_path_static), f"Output file {output_path_static} was not created."

    # Test with new config-based approach
    output_path_animation = 'test_energy_landscape_2d_animation.gif'
    config_anim = PlotConfigs.energy_landscape_2d_animation(
        time_steps_per_second=100,
        fps=20,
        title='Population Coding 2D (Animation)',
        xlabel='State X',
        ylabel='State Y',
        clabel='Activity',
        repeat=True,
        save_path=output_path_animation,
        show=False
    )
    energy_landscape_2d_animation(
        zs_data=us,
        config=config_anim
    )
    assert os.path.isfile(output_path_animation), f"Output file {output_path_animation} was not created."

def test_raster_plot():
    bm.set_dt(dt=0.1)
    cann = CANN1D(num=32)

    task_st = SmoothTracking1D(
        cann_instance=cann,
        Iext=(1., 0.75, 2., 1.75, 3.),
        duration=(10., 10., 10., 10.),
        time_step=bm.get_dt(),
    )
    task_st.get_data()

    def run_step(t, inputs):
        cann(inputs)
        return cann.u.value, cann.r.value

    us, rs = bm.for_loop(
        run_step,
        (
            task_st.run_steps,
            task_st.data
        ),
    )
    spike_trains = firing_rate_to_spike_train(normalize_firing_rates(rs), dt_rate=0.1, dt_spike=0.1)

    # Test with new config-based approach
    output_path = 'test_raster_plot.png'
    config = PlotConfigs.raster_plot(
        title='Raster Plot',
        xlabel='Time',
        ylabel='Neuron Index',
        save_path=output_path,
        show=False
    )
    raster_plot(
        spike_train=spike_trains,
        config=config
    )
    assert os.path.isfile(output_path), f"Output file {output_path} was not created."

def test_average_firing_rate():
    bm.set_dt(dt=0.1)
    cann = CANN1D(num=32)

    task_pc = PopulationCoding1D(
        cann_instance=cann,
        before_duration=10.,
        after_duration=10.,
        duration=20.,
        Iext=0.,
        time_step=bm.get_dt(),
    )
    task_pc.get_data()

    def run_step(t, inputs):
        cann(inputs)
        return cann.u.value, cann.r.value

    us, rs = bm.for_loop(
        run_step,
        (
            task_pc.run_steps,
            task_pc.data,
        )
    )

    # Test with new config-based approach
    output_path_population = 'test_average_firing_rate_population.png'
    config_pop = PlotConfigs.average_firing_rate_plot(
        mode='population',
        title='Average Firing Rate (Population)',
        save_path=output_path_population,
        show=False
    )
    average_firing_rate_plot(
        spike_train=rs,
        dt=0.1,
        config=config_pop
    )

    output_path_per_neuron = 'test_average_firing_rate_per_neuron.png'
    config_neuron = PlotConfigs.average_firing_rate_plot(
        mode='per_neuron',
        title='Average Firing Rate (Per Neuron)',
        save_path=output_path_per_neuron,
        show=False
    )
    average_firing_rate_plot(
        spike_train=rs,
        dt=0.1,
        config=config_neuron
    )
    assert os.path.isfile(output_path_population), f"Output file {output_path_population} was not created."


def test_tuning_curve():
    bm.set_dt(dt=0.1)
    cann = CANN1D(num=32)

    task_st = SmoothTracking1D(
        cann_instance=cann,
        Iext=(0., 0., np.pi, 2*np.pi),
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
        )
    )

    # Test with new config-based approach
    neuron_indices_to_plot = [0, 8, 16]
    output_path = 'test_tuning_curve.png'
    config = PlotConfigs.tuning_curve(
        num_bins=50,
        pref_stim=cann.x,
        title='Tuning Curves of Selected Neurons',
        xlabel='Stimulus Position (rad)',
        ylabel='Average Firing Rate',
        show=False,
        save_path=output_path,
        kwargs={'linewidth': 2, 'marker': 'o', 'markersize': 4}
    )
    tuning_curve(
        stimulus=task_st.Iext_sequence.squeeze(),
        firing_rates=rs,
        neuron_indices=neuron_indices_to_plot,
        config=config
    )
    assert os.path.isfile(output_path), f"Output file {output_path} was not created."