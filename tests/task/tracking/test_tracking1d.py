import brainstate
import braintools

from canns.task.tracking import PopulationCoding1D, TemplateMatching1D, SmoothTracking1D
from canns.models.basic import CANN1D


def test_population_coding_1d():
    brainstate.environ.set(dt=0.1)
    cann = CANN1D(num=512)
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
    # braintools.visualize.animate_1D(
    #     dynamical_vars=[{'ys': us, 'xs': cann.x, 'legend': 'u'},
    #                     {'ys': inps, 'xs': cann.x, 'legend': 'Iext'}],
    #     frame_step=5,
    #     frame_delay=5,
    #     save_path='test_tracking1d_population_coding.gif',
    # )

def test_template_matching_1d():
    brainstate.environ.set(dt=0.1)
    cann = CANN1D(num=512)
    cann.init_state()

    task_tm = TemplateMatching1D(
        cann_instance=cann,
        Iext=0.,
        duration=20.,
        time_step=brainstate.environ.get_dt(),
    )
    task_tm.get_data()

    def run_step(t, inputs):
        cann(inputs)
        return cann.u.value, cann.inp.value

    us, inps = brainstate.compile.for_loop(run_step, task_tm.run_steps, task_tm.data, pbar=brainstate.compile.ProgressBar(10))
    # braintools.visualize.animate_1D(
    #     dynamical_vars=[{'ys': us, 'xs': cann.x, 'legend': 'u'},
    #                     {'ys': inps, 'xs': cann.x, 'legend': 'Iext'}],
    #     frame_step=5,
    #     frame_delay=5,
    #     save_path='test_template_matching.gif',
    # )

def test_smooth_tracking_1d():
    brainstate.environ.set(dt=0.1)
    cann = CANN1D(num=512)
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
        return cann.u.value, cann.inp.value

    us, inps = brainstate.compile.for_loop(run_step, task_st.run_steps, task_st.data, pbar=brainstate.compile.ProgressBar(10))
    # braintools.visualize.animate_1D(
    #     dynamical_vars=[{'ys': us, 'xs': cann.x, 'legend': 'u'},
    #                     {'ys': inps, 'xs': cann.x, 'legend': 'Iext'}],
    #     frame_step=5,
    #     frame_delay=5,
    #     save_path='test_smooth_tracking.gif',
    # )