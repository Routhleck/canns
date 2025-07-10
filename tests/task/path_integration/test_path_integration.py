import brainstate
import brainunit as u
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
    trajectory = task_pi.generate_trajectory()

    hierarchical_net = HierarchicalNetwork(num_module=5, num_place=30)

    def initialize(t, input_stre):
        hierarchical_net(
            velocity = u.math.zeros(2,),
            loc=trajectory.position[0],
            loc_input_stre=input_stre,
        )

    init_time = 500

