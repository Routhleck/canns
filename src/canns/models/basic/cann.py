from __future__ import annotations

import brainpy.math as bm
import jax
from matplotlib import pyplot as plt

from ...typing import time_type
from ._base import BasicModel
from .accel import (
    ACCL_DEFAULT_K,
    ACCL_MODES,
    _pick_k_for_err_target,  # noqa: F401  (re-exported for backward compat)
    make_irec_backend,
)

__all__ = [
    # Base Model
    "BaseCANN",
    "BaseCANN1D",
    "BaseCANN2D",
    # CANN 1D Models
    "CANN1D",
    "CANN1D_SFA",
    # CANN 2D Models
    "CANN2D",
    "CANN2D_SFA",
    # Acceleration modes
    "ACCL_MODES",
    "ACCL_DEFAULT_K",
    # Helper for picking k from the SVD spectrum
    "_pick_k_for_err_target",
]


class BaseCANN(BasicModel):
    """
    Base class for Continuous Attractor Neural Network (CANN) models.
    This class sets up the fundamental properties of the network, including
    neuronal properties, feature space, and the connectivity matrix, which
    are shared by different CANN model variations.
    """

    def __init__(
        self,
        shape: int | tuple[int, ...],
        **kwargs,
    ):
        """
        Initializes the base CANN model.

        Args:
            shape (int or tuple): The number of neurons in the network. If an int is provided,
                                  it will be converted to a single-element tuple. If a tuple is provided,
                                  it defines the shape of the network (e.g., (length, length) for 2D).
                                  Internally, shape is always stored as a tuple.
            **kwargs: Additional keyword arguments passed to the parent BasicModel.
        """
        if isinstance(shape, int):
            self.shape = (shape,)
        elif isinstance(shape, tuple):
            self.shape = shape
        else:
            raise TypeError("shape must be an int or a tuple of ints")
        super().__init__(**kwargs)

    def make_conn(self):
        """
        Constructs the connectivity matrix for the CANN model.
        This method should be implemented in subclasses to define how neurons
        are connected based on their feature preferences.

        Returns:
            Array: A connectivity matrix defining the synaptic strengths between neurons.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    def get_stimulus_by_pos(self, pos):
        """
        Generates an external stimulus based on a given position in the feature space.
        This method should be implemented in subclasses to define how the stimulus is shaped.

        Args:
            pos (float or Array): The position in the feature space where the stimulus is centered.

        Returns:
            Array: An array of stimulus values for each neuron.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")


class BaseCANN1D(BaseCANN):
    """Base class for 1D Continuous Attractor Neural Network (CANN) models.

    It builds the 1D feature space, connectivity kernel, and stimulus helpers
    shared by 1D CANN variants.

    Examples:
        >>> import brainpy.math as bm
        >>> from canns.models.basic.cann import BaseCANN1D
        >>>
        >>> bm.set_dt(0.1)
        >>> model = BaseCANN1D(num=64)
        >>> stimulus = model.get_stimulus_by_pos(0.0)
        >>> stimulus.shape
        (64,)
    """

    def __init__(
        self,
        num: int,
        tau: time_type = 1.0,
        k: float = 8.1,
        a: float = 0.5,
        A: float = 10,
        J0: float = 4.0,
        z_min: float = -bm.pi,
        z_max: float = bm.pi,
        accl_mode: str = "normal",
        accl_k: int | None = None,
        accl_target_err_mrad: float = 5.0,
        **kwargs,
    ):
        """
        Initializes the base 1D CANN model.

        Args:
            num (int): The number of neurons in the network.
            tau (float): The synaptic time constant, controlling how quickly the membrane potential changes.
            k (float): A parameter controlling the strength of the global inhibition.
            a (float): The half-width of the excitatory connection range. It defines the "spread" of local connections.
            A (float): The magnitude (amplitude) of the external stimulus.
            J0 (float): The maximum connection strength between neurons.
            z_min (float): The minimum value of the feature space (e.g., -pi for an angle).
            z_max (float): The maximum value of the feature space (e.g., +pi for an angle).
            accl_mode (str, optional): Acceleration mode for the recurrent matvec.
                One of ``"normal"`` (full rank, default), ``"fast"`` (low-rank
                with rank ``ACCL_DEFAULT_K[("CANN1D", "fast")] = 8``),
                ``"ultra-fast"`` (low-rank with rank 1), ``"auto"`` (pick
                the rank from the SVD spectrum to satisfy
                ``accl_target_err_mrad``), or ``"fft"`` (exact circulant
                matvec, O(n log n)). The recommended ``"fast"``
                setting keeps the bump-position error below ~5 mrad and
                gives 30–245× matvec speedup at ``num >= 512``. The
                ``"fft"`` mode gives a 25–50× speedup **and is exact**
                (rel err < 1e-5) but only when the grid is uniform with
                ``endpoint=False`` (i.e. a clean ring); the canns
                default ``endpoint=True`` grid is not circulant, so the
                FFT path silently falls back to ``"normal"`` with a
                warning. To get the FFT speedup, override the grid:
                ``model.x = bm.linspace(-bm.pi, bm.pi, num,
                endpoint=False)``.
            accl_k (int, optional): Explicit low-rank truncation. If given,
                overrides the default rank implied by ``accl_mode`` (and,
                for ``"auto"``, the spectrum-based auto-pick). Must be
                >= 1. Ignored when ``accl_mode == "normal"``.
            accl_target_err_mrad (float, optional): Maximum allowed
                bump-position error in milliradians, used only when
                ``accl_mode == "auto"`` (and ``accl_k is None``).
                Default ``5.0`` mrad. See
                :func:`_pick_k_for_err_target` for the calibration.
            **kwargs: Additional keyword arguments passed to the parent BasicModel.

        See Also:
            :mod:`canns.models.basic.accel`: the strategy-pattern
            backends that implement the modes above, with a "which
            mode should I use?" decision table at the top of the
            module.
        """
        super().__init__(num, **kwargs)

        # --- Model Parameters ---
        self.tau = tau  # Synaptic time constant.
        self.k = k  # Degree of the rescaled inhibition.
        self.a = a  # Half-width of the range of excitatory connections.
        self.A = A  # Magnitude of the external input.
        self.J0 = J0  # Maximum connection value (amplitude of the connectivity kernel).

        # --- Feature Space Properties ---
        self.z_min = z_min  # Minimum of the feature space.
        self.z_max = z_max  # Maximum of the feature space.
        self.z_range = z_max - z_min  # The total range of the feature space.
        # An array representing the preferred feature value for each neuron.
        self.x = bm.linspace(z_min, z_max, num)
        self.rho = num / self.z_range  # The neural density.
        self.dx = self.z_range / num  # The stimulus density

        # --- Connectivity Matrix ---
        # The connection matrix, defining the strength of synapses between all pairs of neurons.
        self.conn_mat = self.make_conn()

        # --- Acceleration (low-rank / circulant-FFT recurrent matvec) ---
        # All pre-computed factors (SVD factors, FFT kernel, dense
        # conn_mat reference) live on ``self.irec_backend``. ``update()``
        # just calls ``self.irec_backend(self.r.value)`` and the
        # backend chooses the right math. Adding a new mode means
        # adding a new backend class in :mod:`.accel` — this file
        # is not touched. See ``benchmarks/canns-accl/lowrank`` for the SVD
        # trade-offs and ``benchmarks/canns-accl/fft`` for the FFT comparison.
        self._accl_target_err_mrad = accl_target_err_mrad
        self.irec_backend = make_irec_backend(
            self,
            family="CANN1D",
            mode=accl_mode,
            dim=1,
            k=accl_k,
            target_err_mrad=accl_target_err_mrad,
        )

    def dist(self, d):
        """
        Calculates the shortest distance between two points in a circular feature space
        with periodic boundary conditions.

        Args:
            d (Array): The difference between two positions.

        Returns:
            Array: The shortest distance, wrapped around the periodic boundary.
        """
        # Apply periodic boundary condition using the remainder.
        d = bm.remainder(d, self.z_range)
        # Ensure the distance is the shortest path (e.g., the distance between 350 and 10 degrees is 20, not 340).
        d = bm.where(d > self.z_range / 2, d - self.z_range, d)
        return d

    def make_conn(self):
        """
        Constructs the connectivity matrix based on a Gaussian-like profile.
        The connection strength between two neurons depends on the distance
        between their preferred feature values in the circular space.

        Returns:
            Array: A (num x num) connectivity matrix.
        """
        # Prepare coordinate arrays to compute pairwise distances.
        x_left = bm.reshape(self.x, (-1, 1))
        x_right = bm.repeat(self.x.reshape((1, -1)), len(self.x), axis=0)
        # Calculate the pairwise distance matrix with periodic boundaries.
        d = self.dist(x_left - x_right)
        # Compute the connection strengths using a Gaussian (normal distribution) function.
        # Neurons with similar feature preferences will have stronger excitatory connections.
        return self.J0 * bm.exp(-0.5 * bm.square(d / self.a)) / (bm.sqrt(2 * bm.pi) * self.a)

    def get_stimulus_by_pos(self, pos):
        """
        Generates a Gaussian-shaped external stimulus centered at a given position.

        Args:
            pos (float): The center position of the stimulus in the feature space.

        Returns:
            Array: An array of stimulus values for each neuron.
        """
        # The stimulus is a "bump" of activity, modeled by a Gaussian function.
        return self.A * bm.exp(-0.25 * bm.square(self.dist(self.x - pos) / self.a))

    # ------------------------------------------------------------------
    # Acceleration of the recurrent matvec (delegated to a strategy
    # object on ``self.irec_backend``)
    # ------------------------------------------------------------------
    def _setup_accl(
        self,
        accl_mode: str | None = None,
        accl_k: int | None = None,
        accl_target_err_mrad: float = 5.0,
    ) -> None:
        """(Re)build ``self.irec_backend`` for the current ``conn_mat``.

        The actual SVD / FFT pre-compute lives in
        :func:`canns.models.basic.accel.make_irec_backend`; this
        method is the thin convenience wrapper exposed on the model
        so callers (e.g. the test suite) can re-setup the backend
        after rebuilding ``conn_mat`` in place.

        ``accl_mode=None`` keeps the current mode; ``accl_k=None`` lets
        the factory pick the mode default (or the spectrum pick for
        ``"auto"``). For ``"auto"``, ``accl_target_err_mrad`` is only
        used when ``accl_k is None``.
        """
        new_mode = accl_mode if accl_mode is not None else self.irec_backend.mode
        # Remember the err budget so subsequent ``set_accl_mode``
        # calls (without an explicit target) can carry it across.
        self._accl_target_err_mrad = accl_target_err_mrad
        self.irec_backend = make_irec_backend(
            self,
            family="CANN1D",
            mode=new_mode,
            dim=1,
            k=accl_k,
            target_err_mrad=accl_target_err_mrad,
        )

    def set_accl_mode(
        self,
        mode: str | None = None,
        k: int | None = None,
        target_err_mrad: float | None = None,
    ) -> None:
        """Change the acceleration mode or rank at runtime.

        Args:
            mode: New ``accl_mode`` (one of ``ACCL_MODES``). If ``None``,
                the current mode is kept.
            k: New explicit rank. If ``None``, the default rank for
                the (possibly new) mode is used (or the
                ``target_err_mrad``-driven pick for ``mode="auto"``).
            target_err_mrad: New err budget (mrad) for ``mode="auto"``;
                only used when ``mode="auto"`` and ``k is None``.
                If ``None``, keeps the existing budget (or defaults to
                ``5.0`` if the model has never been in ``"auto"``).

        Examples:
            >>> model = CANN1D(num=512)            # normal, k=-1
            >>> model.set_accl_mode("fast")        # now k=8
            >>> model.set_accl_mode("fast", k=4)   # override to k=4
            >>> model.set_accl_mode("auto")        # pick k for 5 mrad
            >>> model.set_accl_mode("auto", target_err_mrad=0.5)  # tighter budget
            >>> model.set_accl_mode("normal")      # back to full rank
        """
        new_mode = mode if mode is not None else self.irec_backend.mode
        if target_err_mrad is None:
            target_err_mrad = getattr(self, "_accl_target_err_mrad", 5.0)
        self._accl_target_err_mrad = target_err_mrad
        self.irec_backend = make_irec_backend(
            self,
            family="CANN1D",
            mode=new_mode,
            dim=1,
            k=k,
            target_err_mrad=target_err_mrad,
        )

    # -- Public introspection (delegate to the backend) -----------------

    @property
    def accl_mode(self) -> str:
        """The current acceleration mode (one of ``ACCL_MODES``).

        Reflects the strategy currently held on ``self.irec_backend``:
        ``"normal"``, ``"fast"``, ``"ultra-fast"``, ``"auto"``, or
        ``"fft"``. See :mod:`canns.models.basic.accel` for the
        per-mode semantics.
        """
        return self.irec_backend.mode

    @property
    def accl_k(self) -> int:
        """The effective low-rank rank, or ``-1`` for full-rank / exact FFT.

        ``-1`` is returned in three cases: dense (``"normal"`` mode),
        circulant-FFT (``"fft"`` mode, which is exact not truncated),
        and ``"auto"`` mode when the budget is too tight to satisfy
        with any lowrank (the model silently falls back to dense).
        """
        return self.irec_backend.k

    @property
    def is_accelerated(self) -> bool:
        """True iff the model is using a non-dense recurrent matvec
        (low-rank or circulant-FFT)."""
        return self.irec_backend.mode != "normal"

    # -- Legacy introspection attributes (used by tests and benchmarks) --

    @property
    def _U_l(self) -> jax.Array | None:
        """The ``U`` factor of the balanced SVD truncation, or ``None``.

        Satisfies ``_U_l @ _V_l.T ≈ conn_mat`` up to the spectral tail
        past ``accl_k``. ``None`` whenever the current backend is not
        :class:`LowRankIrec` (i.e. dense or circulant-FFT). Exposed as
        a read-only attribute for tests and benchmarks that need to
        inspect the low-rank factors directly.
        """
        return self.irec_backend.U

    @property
    def _V_l(self) -> jax.Array | None:
        """The ``V`` factor of the balanced SVD truncation, or ``None``.

        See :attr:`_U_l` for the approximation guarantee and when this
        is ``None``.
        """
        return self.irec_backend.V

    @property
    def _K_fft(self) -> jax.Array | None:
        """The precomputed FFT of the (doubly-)circulant kernel, or ``None``.

        Used by the ``"fft"`` mode to evaluate
        ``real(ifft(K_fft ⊙ fft(r)))`` (1D) or
        ``real(ifft2(K_fft ⊙ fft2(r_2d)))`` (2D). ``None`` whenever
        the current backend is not :class:`CirculantFFTIrec1D` /
        :class:`CirculantFFTIrec2D`. Exposed for tests and benchmarks.
        """
        return self.irec_backend.K_fft

    def _accel_Irec(self, r: jax.Array) -> jax.Array:
        """Compute the recurrent input ``Irec`` for a flat rate vector ``r``.

        Thin wrapper around ``self.irec_backend(r)`` kept for
        backward compatibility (the test suite calls this directly).
        The dispatch (dense vs low-rank vs circulant-FFT) is entirely
        inside the backend object; see :mod:`canns.models.basic.accel`.
        """
        return self.irec_backend(r)


class CANN1D(BaseCANN1D):
    """Standard 1D Continuous Attractor Neural Network (CANN) model.

    This model sustains a localized "bump" of activity that can be driven by
    external input.

    Examples:
        >>> import brainpy.math as bm
        >>> from canns.models.basic import CANN1D
        >>>
        >>> bm.set_dt(0.1)
        >>> model = CANN1D(num=64)
        >>> stimulus = model.get_stimulus_by_pos(0.0)
        >>> model.update(stimulus)
        >>> model.r.value.shape
        (64,)

    Reference:
        Wu, S., Hamaguchi, K., & Amari, S. I. (2008). Dynamics and computation of continuous attractors.
        Neural computation, 20(4), 994-1025.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the 1D CANN model.

        Args:
            (Parameters are inherited from BaseCANN1D)
        """
        super().__init__(*args, **kwargs)

        # Firing rate of the neurons.
        self.r = bm.Variable(bm.zeros(self.shape))
        # Synaptic input to the neurons.
        self.u = bm.Variable(bm.zeros(self.shape))

        # --- Inputs ---
        # External input to the network.
        self.inp = bm.Variable(bm.zeros(self.shape))

    def update(self, inp):
        """Advance the network by one time step.

        Args:
            inp (Array): External input vector of shape ``(num,)``.

        Returns:
            None
        """
        self.inp.value = inp
        # The numerator for the firing rate calculation (a non-linear activation function).
        r1 = bm.square(self.u.value)
        # The denominator, which implements global divisive inhibition.
        r2 = 1.0 + self.k * bm.sum(r1)
        # Calculate the firing rate of each neuron using divisive normalization.
        self.r.value = r1 / r2
        # Calculate the recurrent input from other neurons in the network.
        # In ``accl_mode != "normal"`` this is a low-rank matvec
        # ``U_l @ (V_l.T @ r)`` instead of ``conn @ r`` — see
        # ``BaseCANN1D._accel_Irec`` and ``benchmarks/canns-accl/lowrank``.
        Irec = self._accel_Irec(self.r.value)
        # Update the synaptic inputs using Euler's method. The change depends on a leak
        # current (-u), recurrent input (Irec), and external input (inp).
        self.u.value += (-self.u.value + Irec + self.inp.value) / self.tau * bm.get_dt()


class CANN1D_SFA(BaseCANN1D):
    """1D CANN model with spike-frequency adaptation (SFA).

    SFA adds a slow negative feedback term that can create anticipative tracking
    under sustained inputs.

    Examples:
        >>> import brainpy.math as bm
        >>> from canns.models.basic import CANN1D_SFA
        >>>
        >>> bm.set_dt(0.1)
        >>> model = CANN1D_SFA(num=64)
        >>> stimulus = model.get_stimulus_by_pos(0.0)
        >>> model.update(stimulus)
        >>> model.r.value.shape
        (64,)

    Reference:
        Mi, Y., Fung, C. C., Wong, K. Y., & Wu, S. (2014). Spike frequency adaptation
        implements anticipative tracking in continuous attractor neural networks.
        Advances in neural information processing systems, 27.
    """

    def __init__(
        self,
        num: int,
        tau: time_type = 1.0,
        tau_v: time_type = 50.0,
        k: float = 8.1,
        a: float = 0.3,
        A: float = 0.2,
        J0: float = 1.0,
        z_min: float = -bm.pi,
        z_max: float = bm.pi,
        m: float = 0.3,
        **kwargs,
    ):
        """
        Initializes the 1D CANN model with SFA.

        Args:
            tau_v (float): The time constant for the adaptation variable 'v'. A larger value means slower adaptation.
            m (float): The strength of the adaptation, coupling the membrane potential 'u' to the adaptation variable 'v'.
            (Other parameters are inherited from BaseCANN1D)
        """
        super().__init__(num, tau, k, a, A, J0, z_min, z_max, **kwargs)
        # --- SFA-specific Parameters ---
        self.tau_v = tau_v  # Time constant of the adaptation variable.
        self.m = m  # Strength of the adaptation.

        self.r = bm.Variable(bm.zeros(self.shape))  # Firing rate.
        self.u = bm.Variable(bm.zeros(self.shape))  # Synaptic inputs.
        # self.v: The adaptation variable, which tracks the synaptic inputs 'u' and provides negative feedback.
        self.v = bm.Variable(bm.zeros(self.shape))

        # --- Inputs ---
        self.inp = bm.Variable(bm.zeros(self.shape))  # External input.

    def update(self, inp):
        """Advance the network by one time step with adaptation.

        Args:
            inp (Array): External input vector of shape ``(num,)``.

        Returns:
            None
        """
        self.inp.value = inp
        # Firing rate calculation is the same as the standard CANN model.
        r1 = bm.square(self.u.value)
        r2 = 1.0 + self.k * bm.sum(r1)
        self.r.value = r1 / r2
        # Calculate recurrent input. The low-rank form (``accl_mode !=
        # "normal"``) replaces the dense ``conn @ r`` with
        # ``U_l @ (V_l.T @ r)`` — same accuracy / speedup as CANN1D.
        Irec = self._accel_Irec(self.r.value)
        # Update the synaptic input. Note the additional '- self.v.value' term,
        self.u.value += (
            (-self.u.value + Irec + self.inp.value - self.v.value) / self.tau * bm.get_dt()
        )
        # Update the adaptation variable 'v'. It slowly tracks the membrane potential 'u'
        # and has its own decay, creating a slow negative feedback loop.
        self.v.value += (-self.v.value + self.m * self.u.value) / self.tau_v * bm.get_dt()


class BaseCANN2D(BaseCANN):
    """Base class for 2D Continuous Attractor Neural Network (CANN) models.

    It builds the 2D feature space, connectivity kernel, and stimulus helpers
    shared by 2D CANN variants.

    Examples:
        >>> import brainpy.math as bm
        >>> from canns.models.basic.cann import BaseCANN2D
        >>>
        >>> bm.set_dt(0.1)
        >>> model = BaseCANN2D(length=16)
        >>> stimulus = model.get_stimulus_by_pos([0.0, 0.0])
        >>> stimulus.shape
        (16, 16)
    """

    def __init__(
        self,
        length: int,
        tau: float = 1.0,
        k: float = 8.1,
        a: float = 0.5,
        A: float = 10,
        J0: float = 4.0,
        z_min: float = -bm.pi,
        z_max: float = bm.pi,
        accl_mode: str = "normal",
        accl_k: int | None = None,
        accl_target_err_mrad: float = 5.0,
        **kwargs,
    ):
        """
        Initializes the base 2D CANN model.

        Args:
            length (int): The number of neurons in one dimension of the network (the network is square).
            tau (float): The synaptic time constant, controlling how quickly the membrane potential changes.
            k (float): A parameter controlling the strength of the global inhibition.
            a (float): The half-width of the excitatory connection range. It defines the "spread" of local connections.
            A (float): The magnitude (amplitude) of the external stimulus.
            J0 (float): The maximum connection strength between neurons.
            z_min (float): The minimum value of the feature space (e.g., -pi for an angle).
            z_max (float): The maximum value of the feature space (e.g., +pi for an angle).
            accl_mode (str, optional): Acceleration mode for the recurrent matvec.
                One of ``"normal"`` (full rank, default), ``"fast"`` (low-rank
                with rank ``ACCL_DEFAULT_K[("CANN2D", "fast")] = 32``),
                ``"ultra-fast"`` (low-rank with rank 4), ``"auto"`` (pick
                the rank from the SVD spectrum to satisfy
                ``accl_target_err_mrad``), or ``"fft"`` (exact doubly-
                circulant matvec via 2D FFT, O(L² log L)). At ``length=64``
                the ``"fast"`` setting gives ~70× matvec speedup while
                keeping the bump-position error below ~5 mrad. The
                ``"fft"`` mode gives a 25–50× speedup **and is exact**
                (rel err < 1e-5) but only when the grid is uniform with
                ``endpoint=False`` (i.e. a clean torus); the canns
                default ``endpoint=True`` grid is not doubly-circulant,
                so the FFT path silently falls back to ``"normal"``
                with a warning. To get the FFT speedup, override the
                grid: ``model.x = bm.linspace(-bm.pi, bm.pi, length,
                endpoint=False)``.

                See :mod:`canns.models.basic.accel` for a
                side-by-side "which mode should I use?" guide.
            accl_k (int, optional): Explicit low-rank truncation. If given,
                overrides the default rank implied by ``accl_mode`` (and,
                for ``"auto"``, the spectrum-based auto-pick). Must be
                >= 1. Ignored when ``accl_mode == "normal"``.
            accl_target_err_mrad (float, optional): Maximum allowed
                bump-position error in milliradians, used only when
                ``accl_mode == "auto"`` (and ``accl_k is None``).
                Default ``5.0`` mrad. See
                :func:`_pick_k_for_err_target` for the calibration.
            **kwargs: Additional keyword arguments passed to the parent BasicModel.
        """
        self.length = length
        super().__init__((self.length,) * 2, **kwargs)  # square network of neurons

        # --- Model Parameters ---
        self.tau = tau  # Synaptic time constant.
        self.k = k  # Degree of the rescaled inhibition.
        self.a = a  # Half-width of the range of excitatory connections.
        self.A = A  # Magnitude of the external input.
        self.J0 = J0  # Maximum connection value (amplitude of the connectivity kernel).

        # --- Feature Space Properties ---
        self.z_min = z_min  # Minimum of the feature space.
        self.z_max = z_max  # Maximum of the feature space.
        self.z_range = z_max - z_min  # The total range of the feature space.
        # An array representing the preferred feature value for each neuron.
        self.x = bm.linspace(z_min, z_max, length)
        self.rho = length / self.z_range  # The neural density.
        self.dx = self.z_range / length  # The stimulus density

        # --- Connectivity Matrix ---
        # The connection matrix, defining the strength of synapses between all pairs of neurons.
        self.conn_mat = self.make_conn()

        # --- Acceleration (low-rank / circulant-FFT recurrent matvec) ---
        # All pre-computed factors (SVD factors, 2D FFT kernel, dense
        # conn_mat reference) live on ``self.irec_backend``. ``update()``
        # just calls ``self.irec_backend(self.r.value.reshape(-1))`` and
        # the backend chooses the right math. Adding a new mode means
        # adding a new backend class in :mod:`.accel` — this file is
        # not touched. See ``benchmarks/canns-accl/lowrank`` for the SVD
        # trade-offs and ``benchmarks/canns-accl/fft`` for the FFT comparison.
        self._accl_target_err_mrad = accl_target_err_mrad
        self.irec_backend = make_irec_backend(
            self,
            family="CANN2D",
            mode=accl_mode,
            dim=2,
            length=self.length,
            k=accl_k,
            target_err_mrad=accl_target_err_mrad,
        )

    def show_conn(self):
        """
        Displays the connectivity matrix as an image.
        This method visualizes the connection strengths between neurons in the 2D feature space.
        """
        plt.imshow(bm.as_numpy(self.conn_mat))
        plt.colorbar()
        plt.show()

    def dist(self, d):
        """
        Calculates the shortest distance vector between two points in a 2D feature space
        with periodic boundary conditions (a torus).

        Args:
            d (Array): The difference vector between two positions, e.g., [dx, dy].

        Returns:
            Array: The shortest distance vector, with each component wrapped around
                   the periodic boundary.
        """
        # Define the size of the periodic box for each dimension.
        box_size = bm.asarray([self.z_range, self.z_range])
        # Apply the periodic boundary condition to each component of the vector
        # using the remainder. This wraps the differences into the [0, box_size) interval.
        d = bm.remainder(d, box_size)
        # Ensure each component of the distance vector is the shortest path.
        # For example, in a dimension of size 360, the distance between 350 and 10
        # should be -20 (magnitude 20), not 340.
        d = bm.where(d > box_size / 2, d - box_size, d)
        return d

    def make_conn(self):
        """
        Constructs the connectivity matrix for a 2D grid of neurons based on a
        Gaussian profile. The connection strength between two neurons depends on the
        Euclidean distance between their preferred feature coordinates in a 2D
        toroidal space (space with periodic boundaries in both dimensions).

        Returns:
            Array: A ((num*num) x (num*num)) connectivity matrix.
        """
        # Create a 2D grid of coordinates for all neurons.
        x1, x2 = bm.meshgrid(self.x, self.x)
        # Reshape the grid into a list of coordinate pairs.
        all_coords = bm.stack([x1.flatten(), x2.flatten()]).T

        # Define a function to compute connectivity from one neuron to all others.
        @jax.vmap
        def get_conn_for_one_neuron(source_coord):
            # Calculate the difference vectors from this source neuron to all other neurons.
            # The self.dist function handles the periodic boundary conditions for each vector component.
            diff_vectors = self.dist(source_coord - all_coords)

            # Calculate the scalar Euclidean distance (L2 norm) for each difference vector.
            # This gives the true shortest distance in the 2D toroidal space.
            scalar_distances = bm.linalg.norm(diff_vectors, axis=1)

            # Compute connection strengths using the same Gaussian (normal distribution) function.
            # Neurons with closer coordinates will have stronger excitatory connections.
            conn_strengths = (
                self.J0
                * bm.exp(-0.5 * bm.square(scalar_distances / self.a))
                / (bm.sqrt(2 * bm.pi) * self.a)
            )
            return conn_strengths

        return get_conn_for_one_neuron(all_coords)

    def get_stimulus_by_pos(self, pos):
        """
        Generates a Gaussian-shaped external stimulus centered at a given
        coordinate on the 2D neural grid.

        Args:
            pos (Array, tuple): The center coordinate [x, y] of the stimulus
                                in the feature space.

        Returns:
            Array: A 2D array (grid) of stimulus values for each neuron.
        """
        # Validate that the input position is two-dimensional.
        pos = bm.asarray(pos)
        assert pos.shape == (2,), "Input position must be a 2D coordinate, e.g., [x, y]."
        # Create a 2D grid of coordinates for all neurons.
        x1, x2 = bm.meshgrid(self.x, self.x)
        all_coords = bm.stack([x1.flatten(), x2.flatten()]).T
        # Calculate the distance from the stimulus center to every neuron.
        diff_vectors = self.dist(all_coords - pos)
        scalar_distances = bm.linalg.norm(diff_vectors, axis=1)
        # Calculate the stimulus intensity using a Gaussian function.
        stimulus_flat = self.A * bm.exp(-0.25 * bm.square(scalar_distances / self.a))
        # Reshape the flat stimulus array back into a 2D grid.
        num_neurons_per_dim = self.x.shape[0]
        return stimulus_flat.reshape((num_neurons_per_dim, num_neurons_per_dim))

    # ------------------------------------------------------------------
    # Acceleration of the recurrent matvec (delegated to a strategy
    # object on ``self.irec_backend``)
    # ------------------------------------------------------------------
    def _setup_accl(
        self,
        accl_mode: str | None = None,
        accl_k: int | None = None,
        accl_target_err_mrad: float = 5.0,
    ) -> None:
        """(Re)build ``self.irec_backend`` for the current ``conn_mat``.

        The actual SVD / FFT pre-compute lives in
        :func:`canns.models.basic.accel.make_irec_backend`; this
        method is the thin convenience wrapper exposed on the model
        so callers (e.g. the test suite) can re-setup the backend
        after rebuilding ``conn_mat`` in place.

        ``accl_mode=None`` keeps the current mode; ``accl_k=None`` lets
        the factory pick the mode default (or the spectrum pick for
        ``"auto"``). For ``"auto"``, ``accl_target_err_mrad`` is only
        used when ``accl_k is None``.
        """
        new_mode = accl_mode if accl_mode is not None else self.irec_backend.mode
        # Remember the err budget so subsequent ``set_accl_mode``
        # calls (without an explicit target) can carry it across.
        self._accl_target_err_mrad = accl_target_err_mrad
        self.irec_backend = make_irec_backend(
            self,
            family="CANN2D",
            mode=new_mode,
            dim=2,
            length=self.length,
            k=accl_k,
            target_err_mrad=accl_target_err_mrad,
        )

    def set_accl_mode(
        self,
        mode: str | None = None,
        k: int | None = None,
        target_err_mrad: float | None = None,
    ) -> None:
        """Change the acceleration mode or rank at runtime.

        Args:
            mode: New ``accl_mode`` (one of ``ACCL_MODES``). If ``None``,
                the current mode is kept.
            k: New explicit rank. If ``None``, the default rank for
                the (possibly new) mode is used (or the
                ``target_err_mrad``-driven pick for ``mode="auto"``).
            target_err_mrad: New err budget (mrad) for ``mode="auto"``;
                only used when ``mode="auto"`` and ``k is None``.
                If ``None``, keeps the existing budget (or defaults to
                ``5.0`` if the model has never been in ``"auto"``).

        Examples:
            >>> model = CANN2D(length=32)        # normal, k=-1
            >>> model.set_accl_mode("fast")       # now k=32
            >>> model.set_accl_mode("fast", k=8) # override to k=8
            >>> model.set_accl_mode("auto")       # pick k for 5 mrad
            >>> model.set_accl_mode("auto", target_err_mrad=0.5)  # tighter budget
            >>> model.set_accl_mode("normal")    # back to full rank
        """
        new_mode = mode if mode is not None else self.irec_backend.mode
        if target_err_mrad is None:
            target_err_mrad = getattr(self, "_accl_target_err_mrad", 5.0)
        self._accl_target_err_mrad = target_err_mrad
        self.irec_backend = make_irec_backend(
            self,
            family="CANN2D",
            mode=new_mode,
            dim=2,
            length=self.length,
            k=k,
            target_err_mrad=target_err_mrad,
        )

    # -- Public introspection (delegate to the backend) -----------------

    @property
    def accl_mode(self) -> str:
        """The current acceleration mode (one of ``ACCL_MODES``).

        Reflects the strategy currently held on ``self.irec_backend``:
        ``"normal"``, ``"fast"``, ``"ultra-fast"``, ``"auto"``, or
        ``"fft"``. See :mod:`canns.models.basic.accel` for the
        per-mode semantics.
        """
        return self.irec_backend.mode

    @property
    def accl_k(self) -> int:
        """The effective low-rank rank, or ``-1`` for full-rank / exact FFT.

        ``-1`` is returned in three cases: dense (``"normal"`` mode),
        circulant-FFT (``"fft"`` mode, which is exact not truncated),
        and ``"auto"`` mode when the budget is too tight to satisfy
        with any lowrank (the model silently falls back to dense).
        """
        return self.irec_backend.k

    @property
    def is_accelerated(self) -> bool:
        """True iff the model is using a non-dense recurrent matvec
        (low-rank or circulant-FFT)."""
        return self.irec_backend.mode != "normal"

    # -- Legacy introspection attributes (used by tests and benchmarks) --

    @property
    def _U_l(self) -> jax.Array | None:
        """The ``U`` factor of the balanced SVD truncation, or ``None``.

        Satisfies ``_U_l @ _V_l.T ≈ conn_mat`` up to the spectral tail
        past ``accl_k``. ``None`` whenever the current backend is not
        :class:`canns.models.basic.accel.LowRankIrec` (i.e. dense
        or circulant-FFT). Exposed as a read-only attribute for tests
        and benchmarks that need to inspect the low-rank factors
        directly.
        """
        return self.irec_backend.U

    @property
    def _V_l(self) -> jax.Array | None:
        """The ``V`` factor of the balanced SVD truncation, or ``None``.

        See :attr:`_U_l` for the approximation guarantee and when this
        is ``None``.
        """
        return self.irec_backend.V

    @property
    def _K_fft(self) -> jax.Array | None:
        """The precomputed 2D FFT of the doubly-circulant kernel, or
        ``None``.

        Used by the ``"fft"`` mode to evaluate
        ``real(ifft2(K_fft ⊙ fft2(r_2d))).ravel()`` on a clean torus.
        ``None`` whenever the current backend is not
        :class:`canns.models.basic.accel.CirculantFFTIrec2D`.
        Exposed for tests and benchmarks.
        """
        return self.irec_backend.K_fft

    def _accel_Irec(self, r_flat: jax.Array) -> jax.Array:
        """Compute the recurrent input for a flat rate vector.

        Thin wrapper around ``self.irec_backend(r_flat)`` kept for
        backward compatibility (the test suite calls this directly).
        The dispatch (dense vs low-rank vs doubly-circulant-FFT) is
        entirely inside the backend object; see
        :mod:`canns.models.basic.accel`.

        The input/output are flat ``(length²,)``. The caller is
        responsible for reshaping the result back to a 2D grid if
        needed (see ``CANN2D.update``).
        """
        return self.irec_backend(r_flat)


class CANN2D(BaseCANN2D):
    """2D Continuous Attractor Neural Network (CANN) model.

    Examples:
        >>> import brainpy.math as bm
        >>> from canns.models.basic import CANN2D
        >>>
        >>> bm.set_dt(0.1)
        >>> model = CANN2D(length=16)
        >>> stimulus = model.get_stimulus_by_pos([0.0, 0.0])
        >>> model.update(stimulus)
        >>> model.r.value.shape
        (16, 16)

    Reference:
        Wu, S., Hamaguchi, K., & Amari, S. I. (2008). Dynamics and computation of continuous attractors.
        Neural computation, 20(4), 994-1025.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the 2D CANN model.

        Args:
            (Parameters are inherited from BaseCANN2D)
        """
        super().__init__(*args, **kwargs)

        # Firing rate of the neurons.
        self.r = bm.Variable(bm.zeros((self.length, self.length)))
        # Synaptic input to the neurons.
        self.u = bm.Variable(bm.zeros((self.length, self.length)))

        # --- Inputs ---
        # External input to the neurons
        self.inp = bm.Variable(bm.zeros((self.length, self.length)))

    def update(self, inp):
        """Advance the network by one time step.

        Args:
            inp (Array): External input grid of shape ``(length, length)``.

        Returns:
            None
        """
        self.inp.value = inp
        # The numerator for the firing rate calculation (a non-linear activation function).
        r1 = bm.square(self.u.value)
        # The denominator, which implements global divisive inhibition.
        r2 = 1.0 + self.k * bm.sum(r1)
        # Calculate the firing rate of each neuron using divisive normalization.
        self.r.value = r1 / r2
        # Calculate the recurrent input from other neurons in the network.
        # The original CANN2D uses the row-matvec form
        # ``r.flatten() @ conn``; here we use the equivalent column form
        # ``conn @ r.flatten()`` (same result for the symmetric
        # Gaussian kernel) which fits the low-rank path
        # ``U_l @ (V_l.T @ r)`` uniformly with CANN1D.
        Irec_flat = self._accel_Irec(self.r.value.reshape(-1))
        Irec = Irec_flat.reshape((self.length, self.length))
        # Update the synaptic input based on the recurrent input and external input.
        self.u.value += (-self.u.value + Irec + self.inp.value) / self.tau * bm.get_dt()


class CANN2D_SFA(BaseCANN2D):
    """2D CANN model with spike-frequency adaptation (SFA) dynamics.

    Examples:
        >>> import brainpy.math as bm
        >>> from canns.models.basic import CANN2D_SFA
        >>>
        >>> bm.set_dt(0.1)
        >>> model = CANN2D_SFA(length=16)
        >>> stimulus = model.get_stimulus_by_pos([0.0, 0.0])
        >>> model.update(stimulus)
        >>> model.r.value.shape
        (16, 16)
    """

    def __init__(
        self,
        length: int,
        tau: float = 1.0,
        tau_v: float = 50.0,
        k: float = 8.1,
        a: float = 0.3,
        A: float = 0.2,
        J0: float = 1.0,
        z_min: float = -bm.pi,
        z_max: float = bm.pi,
        m: float = 0.3,
        **kwargs,
    ):
        """
        Initializes the 2D CANN model with SFA dynamics.
        """
        super().__init__(
            length=length, tau=tau, k=k, a=a, A=A, J0=J0, z_min=z_min, z_max=z_max, **kwargs
        )
        # --- SFA-specific Parameters ---
        self.tau_v = tau_v  # Time Constant of the adaptation variable.
        self.m = m  # Strength of the adaptation.

        self.r = bm.Variable(bm.zeros((self.length, self.length)))  # Firing rate.
        self.u = bm.Variable(bm.zeros((self.length, self.length)))  # Synaptic input.
        # self.v: The adaptation variable, which tracks the synaptic inputs 'u' and provides negative feedback.
        self.v = bm.Variable(bm.zeros((self.length, self.length)))

        # --- Inputs ---
        self.inp = bm.Variable(bm.zeros((self.length, self.length)))  # External input.

    def update(self, inp):
        """Advance the network by one time step with adaptation.

        Args:
            inp (Array): External input grid of shape ``(length, length)``.

        Returns:
            None
        """
        self.inp.value = inp
        # Firing rate calculation is the same as the standard CANN model.
        r1 = bm.square(self.u.value)
        r2 = 1.0 + self.k * bm.sum(r1)
        self.r.value = r1 / r2
        # Calculate recurrent input. The low-rank form (``accl_mode !=
        # "normal"``) replaces the dense matvec with
        # ``U_l @ (V_l.T @ r)`` — same accuracy / speedup as CANN2D.
        Irec_flat = self._accel_Irec(self.r.value.reshape(-1))
        Irec = Irec_flat.reshape((self.length, self.length))
        # Update the synaptic input. Note the additional '- self.v.value' term,
        self.u.value += (
            (-self.u.value + Irec + self.inp.value - self.v.value) / self.tau * bm.get_dt()
        )
        # Update the adaptation variable 'v'. It slowly tracks the membrane potential 'u'
        # and has its own decay, creating a slow negative feedback loop.
        self.v.value += (-self.v.value + self.m * self.u.value) / self.tau_v * bm.get_dt()
