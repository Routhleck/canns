import brainpy as bp
import brainpy.math as bm

from canns.analyzer.visualization import energy_landscape_1d_animation
from canns.task.tracking import PopulationCoding1D, TemplateMatching1D, SmoothTracking1D
from canns.models.basic import CANN1D


def test_population_coding_1d():
    bm.set_dt(dt=0.1)
    cann = CANN1D(num=512)

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
            task_pc.data,
        ),
    )

    # energy_landscape_1d_animation(
    #     {'u': (cann.x, us), 'Iext': (cann.x, inps)},
    #     time_steps_per_second=100,
    #     fps=20,
    #     title='Population Coding 1D',
    #     xlabel='State',
    #     ylabel='Activity',
    #     repeat=True,
    #     save_path='test_population_coding_1d.gif',
    #     show=False,
    # )

def test_template_matching_1d():
    bm.set_dt(dt=0.1)
    cann = CANN1D(num=512)

    task_tm = TemplateMatching1D(
        cann_instance=cann,
        Iext=0.,
        duration=20.,
        time_step=bm.get_dt(),
    )
    task_tm.get_data()

    def run_step(t, inputs):
        cann(inputs)
        return cann.u.value, cann.inp.value

    us, inps = bm.for_loop(
        run_step,
        (
            task_tm.run_steps,
            task_tm.data
        )
    )

    # energy_landscape_1d_animation(
    #     {'u': (cann.x, us), 'Iext': (cann.x, inps)},
    #     time_steps_per_second=100,
    #     fps=20,
    #     title='Template Matching 1D',
    #     xlabel='State',
    #     ylabel='Activity',
    #     repeat=True,
    #     save_path='test_template_matching_1d.gif',
    #     show=False,
    # )

def test_smooth_tracking_1d():
    bm.set_dt(dt=0.1)
    cann = CANN1D(num=512)

    task_st = SmoothTracking1D(
        cann_instance=cann,
        Iext=(1., 0.75, 2., 1.75, 3.),
        duration=(10., 10., 10., 10.),
        time_step=bm.get_dt(),
    )
    task_st.get_data()

    def run_step(t, inputs):
        cann(inputs)
        return cann.u.value, cann.inp.value

    us, inps = bm.for_loop(
        run_step,
        (
            task_st.run_steps,
            task_st.data
        )
    )
    # energy_landscape_1d_animation(
    #     {'u': (cann.x, us), 'Iext': (cann.x, inps)},
    #     time_steps_per_second=100,
    #     fps=20,
    #     title='Smooth Tracking 1D',
    #     xlabel='State',
    #     ylabel='Activity',
    #     repeat=True,
    #     save_path='test_smooth_tracking_1d.gif',
    #     show=False,
    # )