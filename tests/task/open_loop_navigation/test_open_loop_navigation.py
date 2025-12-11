import brainpy as bp
import brainpy.math as bm
import brainpy.math as bm
import numpy as np

from canns.models.basic import HierarchicalNetwork
from canns.task.open_loop_navigation import OpenLoopNavigationTask

def test_path_integration():
    bm.set_dt(dt=0.1)
    task_sn = OpenLoopNavigationTask(
        width=5,
        height=5,
        speed_mean=0.04,
        speed_std=0.016,
        duration=100.0,
        dt=0.1,
        start_pos=(2.5, 2.5),
        progress_bar=False,
    )
    task_sn.get_data()
    task_sn.show_data(show=False, save_path='trajectory_test.png')

    hierarchical_net = HierarchicalNetwork(num_module=5, num_place=30)

    def initialize(t, input_stre):
        hierarchical_net(
            velocity = bm.zeros(2,),
            loc=task_sn.data.position[0],
            loc_input_stre=input_stre,
        )

    init_time = 10
    indices = np.arange(init_time)
    input_stre = np.zeros(init_time)
    input_stre[:5] = 100.
    bm.for_loop(
        initialize,
        (
            bm.asarray(indices),
            bm.asarray(input_stre),
        ),
    )

