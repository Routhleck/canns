import brainstate as bst
import brainunit as u

from ._base import BasicModel


class CANN1D(BasicModel):
    """
    A 1D continuous attractor network model.

    Reference:
        @article{wu2008dynamics,
        title={Dynamics and computation of continuous attractors},
        author={Wu, Si and Hamaguchi, Kosuke and Amari, Shun-ichi},
        journal={Neural computation},
        volume={20},
        number={4},
        pages={994--1025},
        year={2008},
        publisher={MIT Press One Rogers Street, Cambridge, MA 02142-1209, USA journals-info~…}
        }

        @article{mi2014spike,
          title={Spike frequency adaptation implements anticipative tracking in continuous attractor neural networks},
          author={Mi, Yuanyuan and Fung, CC and Wong, KY and Wu, Si},
          journal={Advances in neural information processing systems},
          volume={27},
          year={2014}
        }

    """

    def __init__(
        self,
        num,
        tau=1.0,
        tau_v=50.0,
        k=8.1,
        a=0.3,
        A=0.2,
        J0=1.0,
        z_min=-u.math.pi,
        z_max=u.math.pi,
        m=0.3,
        **kwargs,
    ):
        super().__init__(num, **kwargs)

        # parameters
        self.tau = tau  # The synaptic time constant
        self.tau_v = tau_v
        self.k = k  # Degree of the rescaled inhibition
        self.a = a  # Half-width of the range of excitatory connections
        self.A = A  # Magnitude of the external input
        self.J0 = J0  # Maximum connection value
        self.m = m

        # feature space
        self.z_min = z_min
        self.z_max = z_max
        self.z_range = z_max - z_min
        self.x = u.math.linspace(z_min, z_max, num)
        self.rho = num / self.z_range  # The neural density
        self.dx = self.z_range / num  # The stimulus density

        # The connection matrix
        self.conn_mat = self.make_conn()

    def init_state(self, *args, **kwargs):
        # variables
        self.r = bst.HiddenState(u.math.zeros(self.varshape))  # type: ignore[arg-type]
        self.u = bst.HiddenState(u.math.zeros(self.varshape))  # type: ignore[arg-type]
        self.v = bst.HiddenState(u.math.zeros(self.varshape))  # type: ignore[arg-type]

        # inputs
        self.inp = bst.State(u.math.zeros(self.varshape))  # type: ignore[arg-type]

    def dist(self, d):
        d = u.math.remainder(d, self.z_range)
        d = u.math.where(d > self.z_range / 2, d - self.z_range, d)
        return d

    def make_conn(self):
        x_left = u.math.reshape(self.x, (-1, 1))
        x_right = u.math.repeat(self.x.reshape((1, -1)), len(self.x), axis=0)
        d = self.dist(x_left - x_right)
        conn = (
            self.J0
            * u.math.exp(-0.5 * u.math.square(d / self.a))
            / (u.math.sqrt(2 * u.math.pi) * self.a)
        )
        return conn

    def get_stimulus_by_pos(self, pos):
        return self.A * u.math.exp(-0.25 * u.math.square(self.dist(self.x - pos) / self.a))

    def update(self, inp):
        self.inp.value = inp
        r1 = u.math.square(self.u.value)
        r2 = 1.0 + self.k * u.math.sum(r1)
        self.r.value = r1 / r2
        Irec = u.math.dot(self.conn_mat, self.r.value)
        self.u.value += (
            (-self.u.value + Irec + self.inp.value - self.v.value) / self.tau * bst.environ.get_dt()
        )
        self.v.value += (-self.v.value + self.m * self.u.value) / self.tau_v * bst.environ.get_dt()
