import os

import brainstate

from canns.analyzer.visualize import energy_landscape_1d_animation, energy_landscape_2d_animation, \
    energy_landscape_1d_static, energy_landscape_2d_static, raster_plot, average_firing_rate_plot
from canns.analyzer.utils import firing_rate_to_spike_train, normalize_firing_rates
from canns.task.tracking import PopulationCoding1D, PopulationCoding2D, SmoothTracking1D
from canns.models.basic import CANN1D, CANN2D


def test_energy_landscape_1d():
    brainstate.environ.set(dt=0.1)
    cann = CANN1D(num=32)
    cann.init_state()

    task_pc = PopulationCoding1D(
        cann_instance=cann,
        before_duration=10.,
        after_duration=10.,
        duration=20.,
        Iext=0.,
        time_step=brainstate.environ.get_dt(),
    )
    task_pc.get_data()

    def run_step(t, inputs):
        cann(inputs)
        return cann.u.value, cann.inp.value

    us, inps = brainstate.compile.for_loop(run_step, task_pc.run_steps, task_pc.data, pbar=brainstate.compile.ProgressBar(10))

    output_path_static = 'test_energy_landscape_1d_static.gif'
    energy_landscape_1d_static(
        {'u': (cann.x, us[0]), 'Iext': (cann.x, inps[0])},
        title='Population Coding 1D (Static)',
        xlabel='State',
        ylabel='Activity',
        save_path=output_path_static,
        show=False,
    )
    assert os.path.isfile(output_path_static), f"Output file {output_path_static} was not created."

    output_path_animation = 'test_energy_landscape_1d_animation.gif'
    energy_landscape_1d_animation(
        {'u': (cann.x, us), 'Iext': (cann.x, inps)},
        time_steps_per_second=100,
        fps=20,
        title='Population Coding 1D (Animation)',
        xlabel='State',
        ylabel='Activity',
        repeat=True,
        save_path=output_path_animation,
        show=False,
    )
    assert os.path.isfile(output_path_animation), f"Output file {output_path_animation} was not created."

def test_energy_landscape_2d():
    brainstate.environ.set(dt=0.1)
    cann = CANN2D(length=4)
    cann.init_state()

    task_pc = PopulationCoding2D(
        cann_instance=cann,
        before_duration=10.,
        after_duration=10.,
        duration=20.,
        Iext=[0., 0.],
        time_step=brainstate.environ.get_dt(),
    )
    task_pc.get_data()

    def run_step(t, inputs):
        cann(inputs)
        return cann.u.value, cann.r.value, cann.inp.value

    us, rs, inps = brainstate.compile.for_loop(run_step, task_pc.run_steps, task_pc.data, pbar=brainstate.compile.ProgressBar(10))

    output_path_static = 'test_energy_landscape_2d_static.png'
    energy_landscape_2d_static(
        z_data=us[0],
        title='Population Coding 2D (Static)',
        xlabel='State X',
        ylabel='State Y',
        clabel='Activity',
        save_path=output_path_static,
        show=False,
    )
    assert os.path.isfile(output_path_static), f"Output file {output_path_static} was not created."

    output_path_animation = 'test_energy_landscape_2d_animation.gif'
    energy_landscape_2d_animation(
        zs_data=us,
        time_steps_per_second=100,
        fps=20,
        title='Population Coding 2D (Animation)',
        xlabel='State X',
        ylabel='State Y',
        clabel='Activity',
        repeat=True,
        save_path=output_path_animation,
        show=False,
    )
    assert os.path.isfile(output_path_animation), f"Output file {output_path_animation} was not created."

def test_raster_plot():
    brainstate.environ.set(dt=0.1)
    cann = CANN1D(num=32)
    cann.init_state()

    task_st = SmoothTracking1D(
        cann_instance=cann,
        Iext=(1., 0.75, 2., 1.75, 3.),
        duration=(10., 10., 10., 10.),
        time_step=brainstate.environ.get_dt(),
    )
    task_st.get_data()

    def run_step(t, inputs):
        cann(inputs)
        return cann.u.value, cann.r.value

    us, rs = brainstate.compile.for_loop(run_step, task_st.run_steps, task_st.data,
                                           pbar=brainstate.compile.ProgressBar(10))
    spike_trains = firing_rate_to_spike_train(normalize_firing_rates(rs), dt_rate=0.1, dt_spike=0.1)

    output_path = 'test_raster_plot.png'
    raster_plot(
        spike_trains,
        title='Raster Plot',
        xlabel='Time',
        ylabel='Neuron Index',
        save_path=output_path,
        show=False,
    )
    assert os.path.isfile(output_path), f"Output file {output_path} was not created."

def test_average_firing_rate():
    brainstate.environ.set(dt=0.1)
    cann = CANN1D(num=32)
    cann.init_state()

    task_pc = PopulationCoding1D(
        cann_instance=cann,
        before_duration=10.,
        after_duration=10.,
        duration=20.,
        Iext=0.,
        time_step=brainstate.environ.get_dt(),
    )
    task_pc.get_data()

    def run_step(t, inputs):
        cann(inputs)
        return cann.u.value, cann.r.value

    us, rs = brainstate.compile.for_loop(run_step, task_pc.run_steps, task_pc.data,
                                           pbar=brainstate.compile.ProgressBar(10))

    output_path_population = 'test_average_firing_rate_population.png'
    average_firing_rate_plot(
        rs,
        mode='population',
        title='Average Firing Rate',
        dt=0.1,
        save_path=output_path_population,
        show=False,
    )

    output_path_per_neuron = 'test_average_firing_rate_per_neuron.png'
    average_firing_rate_plot(
        rs,
        mode='per_neuron',
        title='Average Firing Rate',
        dt=0.1,
        save_path=output_path_per_neuron,
        show=False,
    )
    assert os.path.isfile(output_path_population), f"Output file {output_path_population} was not created."

