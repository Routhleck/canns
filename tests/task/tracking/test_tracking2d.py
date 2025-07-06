import brainstate
import braintools

from canns.task.tracking import PopulationCoding2D, TemplateMatching2D, SmoothTracking2D
from canns.models import CANN2D, CANN2D_SFA


def test_population_coding_2d():
    brainstate.environ.set(dt=0.1)
    cann = CANN2D(length=100)
    cann.init_state()

    task_pc = PopulationCoding2D(
        cann_instance=cann,
        before_duration=10.,
        after_duration=10.,
        duration=20.,
        Iext=[0., 0.],
        time_step=brainstate.environ.get_dt(),
    )
    task_pc.init_state()

    def run_step(t):
        task_pc()
        cann(task_pc.inputs.value)
        return cann.u.value, cann.r.value, cann.inp.value

    us, rs, inps = brainstate.compile.for_loop(run_step, task_pc.run_steps, pbar=brainstate.compile.ProgressBar(10))
    # braintools.visualize.animate_2D(
    #     values=us.reshape((us.shape[0], -1)),
    #     net_size=(cann.length, cann.length),
    #     dt=brainstate.environ.get_dt(),
    #     frame_step=2,
    #     frame_delay=5,
    #     save_path='test_tracking2d_population_coding.gif',
    # )

def test_template_matching_2d():
    brainstate.environ.set(dt=0.1)
    cann = CANN2D(length=100)
    cann.init_state()

    task_tm = TemplateMatching2D(
        cann_instance=cann,
        Iext=[0., 0.],
        duration=20.,
        time_step=brainstate.environ.get_dt(),
    )
    task_tm.init_state()

    def run_step(t):
        task_tm()
        cann(task_tm.inputs.value)
        return cann.u.value, cann.r.value, cann.inp.value

    us, rs, inps = brainstate.compile.for_loop(run_step, task_tm.run_steps, pbar=brainstate.compile.ProgressBar(10))
    # braintools.visualize.animate_2D(
    #     values=us.reshape((us.shape[0], -1)),
    #     net_size=(cann.length, cann.length),
    #     dt=brainstate.environ.get_dt(),
    #     frame_step=2,
    #     frame_delay=5,
    #     save_path='test_tracking2d_template_matching.gif',
    # )

def test_smooth_tracking_2d():
    brainstate.environ.set(dt=0.1)
    cann = CANN2D_SFA(length=100)
    cann.init_state()

    task_st = SmoothTracking2D(
        cann_instance=cann,
        Iext=([0., 0.], [1., 1.], [0.75, 0.75], [2., 2.], [1.75, 1.75], [3., 3.]),
        duration=(10. ,10., 10., 10., 10.),
        time_step=brainstate.environ.get_dt(),
    )
    task_st.init_state()

    def run_step(t):
        task_st()
        cann(task_st.inputs.value)
        return cann.u.value, cann.r.value, cann.inp.value

    us, rs, inps = brainstate.compile.for_loop(run_step, task_st.run_steps, pbar=brainstate.compile.ProgressBar(10))
    # braintools.visualize.animate_2D(
    #     values=us.reshape((us.shape[0], -1)),
    #     net_size=(cann.length, cann.length),
    #     dt=brainstate.environ.get_dt(),
    #     frame_step=2,
    #     frame_delay=5,
    #     save_path='test_tracking2d_smooth_tracking_sfa.gif',
    # )