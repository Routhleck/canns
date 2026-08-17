import brainpy.math as bm
import numpy as np
import pytest

from canns.models.basic import CANN1D
from canns.task.tracking import PopulationCoding1D, SmoothTracking1D, TemplateMatching1D


def test_population_coding_1d():
    bm.set_dt(dt=0.1)
    cann = CANN1D(num=512)

    task_pc = PopulationCoding1D(
        cann_instance=cann,
        before_duration=10.0,
        after_duration=10.0,
        duration=20.0,
        Iext=0.0,
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
        Iext=0.0,
        duration=20.0,
        time_step=bm.get_dt(),
    )
    task_tm.get_data()

    def run_step(t, inputs):
        cann(inputs)
        return cann.u.value, cann.inp.value

    us, inps = bm.for_loop(run_step, (task_tm.run_steps, task_tm.data))

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


def test_template_matching_1d_noise_level_zero_is_clean():
    """noise_level=0 must produce the exact stimulus at every time step."""
    bm.set_dt(dt=0.1)
    cann = CANN1D(num=64)

    task = TemplateMatching1D(
        cann_instance=cann,
        Iext=0.0,
        duration=2.0,
        time_step=bm.get_dt(),
        noise_level=0.0,
    )
    task.get_data(progress_bar=False)

    stimulus = task.get_stimulus_by_pos(task.Iext_sequence[0])
    # data has shape (T, *network_shape); compare each timestep to stimulus.
    for t in range(task.data.shape[0]):
        np.testing.assert_allclose(task.data[t], stimulus, atol=1e-12)
    # And the across-time variance must be exactly zero (no noise term).
    assert task.data.std(axis=0).max() == 0.0


def test_template_matching_1d_noise_level_scales_std():
    """Doubling noise_level should roughly double the per-pixel noise std."""
    bm.set_dt(dt=0.1)
    cann = CANN1D(num=64)

    np.random.seed(0)
    task_low = TemplateMatching1D(
        cann_instance=cann,
        Iext=0.0,
        duration=10.0,
        time_step=bm.get_dt(),
        noise_level=0.1,
    )
    task_low.get_data(progress_bar=False)

    np.random.seed(0)
    task_high = TemplateMatching1D(
        cann_instance=cann,
        Iext=0.0,
        duration=10.0,
        time_step=bm.get_dt(),
        noise_level=0.3,
    )
    task_high.get_data(progress_bar=False)

    # The per-time-step noise samples are drawn independently, so we compare
    # the std of the *difference* between high-noise and low-noise data:
    # (high - low) = (0.3 - 0.1) * A * randn() = 0.2 * A * randn().
    diff = task_high.data - task_low.data
    expected_std = 0.2 * cann.A
    # Use a loose tolerance: ~5% on a 640-sample empirical std.
    np.testing.assert_allclose(diff.std(), expected_std, rtol=0.10)


def test_template_matching_1d_invalid_noise_level_rejected():
    """Negative noise_level must be rejected at construction time."""
    bm.set_dt(dt=0.1)
    cann = CANN1D(num=64)

    with pytest.raises(ValueError, match="noise_level must be non-negative"):
        TemplateMatching1D(
            cann_instance=cann,
            Iext=0.0,
            duration=1.0,
            time_step=bm.get_dt(),
            noise_level=-0.01,
        )


def test_smooth_tracking_1d():
    bm.set_dt(dt=0.1)
    cann = CANN1D(num=512)

    task_st = SmoothTracking1D(
        cann_instance=cann,
        Iext=(1.0, 0.75, 2.0, 1.75, 3.0),
        duration=(10.0, 10.0, 10.0, 10.0),
        time_step=bm.get_dt(),
    )
    task_st.get_data()

    def run_step(t, inputs):
        cann(inputs)
        return cann.u.value, cann.inp.value

    us, inps = bm.for_loop(run_step, (task_st.run_steps, task_st.data))
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
