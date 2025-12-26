import brainpy as bp
import brainpy.math as bm
import numpy as np

from canns.analyzer.visualization import energy_landscape_2d_animation
from canns.task.tracking import PopulationCoding2D, TemplateMatching2D, SmoothTracking2D
from canns.models.basic import CANN2D, CANN2D_SFA


def test_population_coding_2d():
    bm.set_dt(dt=0.1)
    cann = CANN2D(length=16)

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
    # energy_landscape_2d_animation(
    #     zs_data=us,
    #     time_steps_per_second=100,
    #     fps=20,
    #     title='Population Coding 2D',
    #     xlabel='State X',
    #     ylabel='State Y',
    #     clabel='Activity',
    #     repeat=True,
    #     save_path='test_population_coding_2d.gif',
    #     show=False,
    # )

def test_template_matching_2d():
    bm.set_dt(dt=0.1)
    cann = CANN2D(length=16)

    task_tm = TemplateMatching2D(
        cann_instance=cann,
        Iext=[0., 0.],
        duration=20.,
        time_step=bm.get_dt(),
    )
    task_tm.get_data()

    def run_step(t, inputs):
        cann(inputs)
        return cann.u.value, cann.r.value, cann.inp.value

    us, rs, inps = bm.for_loop(
        run_step,
        (
            task_tm.run_steps,
            task_tm.data
        )
    )
    # energy_landscape_2d_animation(
    #     zs_data=us,
    #     time_steps_per_second=100,
    #     fps=20,
    #     title='Template Matching 2D',
    #     xlabel='State X',
    #     ylabel='State Y',
    #     clabel='Activity',
    #     repeat=True,
    #     save_path='test_template_matching_2d.gif',
    #     show=False,
    # )

def test_smooth_tracking_2d():
    bm.set_dt(dt=0.1)
    cann = CANN2D_SFA(length=16)

    task_st = SmoothTracking2D(
        cann_instance=cann,
        Iext=([0., 0.], [1., 1.], [0.75, 0.75], [2., 2.], [1.75, 1.75], [3., 3.]),
        duration=(10. ,10., 10., 10., 10.),
        time_step=bm.get_dt(),
    )
    task_st.get_data()

    def run_step(t, inputs):
        cann(inputs)
        return cann.u.value, cann.r.value, cann.inp.value

    us, rs, inps = bm.for_loop(
        run_step,
        (
            task_st.run_steps,
            task_st.data
        )
    )
    # energy_landscape_2d_animation(
    #     zs_data=us,
    #     time_steps_per_second=100,
    #     fps=20,
    #     title='Smooth Tracking 2D',
    #     xlabel='State X',
    #     ylabel='State Y',
    #     clabel='Activity',
    #     repeat=True,
    #     save_path='test_smooth_tracking_2d.gif',
    #     show=False,
    # )