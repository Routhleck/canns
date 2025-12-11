import brainpy as bp


class BasicModel(bp.DynamicalSystem):
    """
    Base class for all basic CANN models.

    This class serves as the foundation for implementing Continuous Attractor Neural Network
    (CANN) models in the library. It extends BrainState's Dynamics class to provide a unified
    interface for defining neural network dynamics with state variables, update rules, and
    initialization methods.

    All basic CANN models (CANN1D, CANN2D, hierarchical models, etc.) should inherit from
    this class to ensure consistent behavior and compatibility with the training and
    analysis framework.

    Key Features:
        - Automatic state management through BrainState
        - JAX-compatible for GPU/TPU acceleration
        - Support for compiled execution via brainpy transforms
        - Compatible with visualization and analysis tools

    Expected Subclass Implementation:
        Subclasses should implement the following methods:

        - ``update(inp)``: Define single-step dynamics given external input
        - ``cell_coords()``: Return neuron coordinates in feature space (for CANNs)

    Example:
        >>> import brainpy.math as bm
        >>> from canns.models.basic import CANN1D
        >>>
        >>> # Create a 1D CANN with 512 neurons
        >>> bm.set_dt(0.1)
        >>> model = CANN1D(num=512)
        >>>
        >>> # Run a single update step
        >>> model.update(inp=0.5)

    See Also:
        - :class:`~canns.models.basic.CANN1D`: 1D continuous attractor model
        - :class:`~canns.models.basic.CANN2D`: 2D continuous attractor model
        - :class:`~canns.models.basic.HierarchicalNetwork`: Multi-module hierarchical model
    """

    pass


class BasicModelGroup(bp.DynSysGroup):
    """
    Base class for groups of basic CANN models.

    This class provides infrastructure for managing collections of multiple neural network
    models that work together as a coordinated system. It extends BrainState's DynamicsGroup
    to handle hierarchical or modular architectures where multiple sub-networks interact.

    Use cases include:
        - Hierarchical path integration networks with multiple grid cell modules
        - Multi-scale CANN systems with different spatial resolutions
        - Composite models combining different neural population types
        - Ensemble models for robust computation

    Key Features:
        - Manages multiple sub-model instances
        - Coordinates state initialization across all models
        - Handles sequential or parallel updates of sub-models
        - Aggregates outputs from multiple models
        - Compatible with JAX compilation and GPU acceleration

    Expected Subclass Implementation:
        Subclasses should implement:

        - ``__init__(...)``: Create and register sub-model instances
        - ``update(...)``: Define update logic coordinating all sub-models
        - Custom methods for inter-model communication if needed

    Attributes:
        Sub-models should be registered as instance attributes, which BrainState
        will automatically track and manage.

    Example:
        >>> import brainpy.math as bm
        >>> from canns.models.basic import HierarchicalNetwork
        >>>
        >>> # Create hierarchical network with 4 grid modules and 64 place cells
        >>> bm.set_dt(0.1)
        >>> network = HierarchicalNetwork(num_module=4, num_place=64)
        >>>
        >>> # Update with velocity and position inputs
        >>> velocity = [0.1, 0.2]  # [vx, vy]
        >>> location = [0.5, 0.5]  # [x, y]
        >>> network.update(velocity, location)

    See Also:
        - :class:`~canns.models.basic.HierarchicalNetwork`: Example implementation
        - :class:`~bp.DynSysGroup`: Parent class from BrainPy
    """

    pass
