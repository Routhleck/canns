import brainstate
import brainunit as u
import numpy as np

from canns.models.basic import HierarchicalNetwork
from canns.task.path_integration import PathIntegrationTask

def test_path_integration():
    brainstate.environ.set(dt=0.1)
    task_pi = PathIntegrationTask(
        width=5,
        height=5,
        speed_mean=0.04,
        speed_std=0.016,
        duration=100.0,
        dt=0.1,
        start_pos=(2.5, 2.5),
        progress_bar=False,
    )
    task_pi.get_data()
    task_pi.show_data(show=False, save_path='trajectory_test.png')

    hierarchical_net = HierarchicalNetwork(num_module=5, num_place=30)
    hierarchical_net.init_state()

    def initialize(t, input_stre):
        hierarchical_net(
            velocity = u.math.zeros(2,),
            loc=task_pi.data.position[0],
            loc_input_stre=input_stre,
        )

    init_time = 10
    indices = np.arange(init_time)
    input_stre = np.zeros(init_time)
    input_stre[:5] = 100.
    brainstate.compile.for_loop(
        initialize,
        u.math.asarray(indices),
        u.math.asarray(input_stre),
        pbar=brainstate.compile.ProgressBar(10),
    )

