from ..basic._base import BasicModel, BasicModelGroup


class BrainInspiredModel(BasicModel):
    """
    Base class for brain-inspired neural network models.

    This abstract base class provides a standardized interface for brain-inspired
    models that can work with various learning algorithms (Hebbian, Oja, BCM, STDP,
    etc.) and prediction workflows. It defines conventions for weight parameters,
    state variables, and energy computations that enable seamless integration with
    the trainer ecosystem.

    The class is designed to support both feedforward and recurrent architectures,
    with optional energy-based dynamics and Hebbian-compatible learning.

    Parameters
    ----------
    **kwargs
        Keyword arguments passed to parent BasicModel

    Attributes
    ----------
    weight_attr : str (property)
        Name of the weight parameter attribute (default: "W")
    predict_state_attr : str (property)
        Name of the state vector attribute used for predictions (default: "s")

    Methods
    -------
    apply_hebbian_learning(train_data)
        Model-specific Hebbian learning (optional, override when needed)
    predict(pattern)
        Generate prediction for input pattern (must be implemented)
    energy : float (property)
        Current energy of model state (must be implemented)
    resize(num_neurons, preserve_submatrix=True)
        Optionally resize model dimensions (override when supported)
    
    Notes
    -----
    **Trainer Compatibility:**
    
    For Hebbian-style training:
    - Expose a weight parameter with ``.value`` array of shape (N, N) as ``bm.Variable``
    - Default attribute name is ``W``; override ``weight_attr`` property if different
    - Optional: implement ``apply_hebbian_learning()`` for model-specific behavior
    - Generic trainers can directly update weights without this method
    
    For prediction workflows:
    - Implement ``update(prev_energy)`` method for energy-driven state updates (optional)
    - Implement ``energy`` property returning scalar energy value
    - Expose state vector with ``.value`` as 1D array (default name: ``s``)
    - Override ``predict_state_attr`` if state is stored under different name
    
    For dynamic resizing:
    - Optionally implement ``resize()`` to support dimension changes
    - Useful when training with variable-length patterns

    Example
    -------
    Creating a simple linear model:

    >>> import brainpy.math as bm
    >>> from canns.models.brain_inspired import BrainInspiredModel
    >>> 
    >>> class SimpleLinear(BrainInspiredModel):
    ...     def __init__(self, input_size, output_size):
    ...         super().__init__()
    ...         self.input_size = input_size
    ...         self.output_size = output_size
    ...         # Weight matrix (trainable)
    ...         self.W = bm.Variable(bm.random.randn(output_size, input_size) * 0.01)
    ...         # State vector for predictions
    ...         self.s = bm.Variable(bm.zeros(output_size))
    ...     
    ...     def forward(self, x):
    ...         self.s.value = self.W.value @ x
    ...         return self.s.value
    ...     
    ...     def predict(self, pattern):
    ...         return self.forward(pattern)
    ...     
    ...     @property
    ...     def energy(self):
    ...         # Simple quadratic energy
    ...         return -0.5 * bm.sum(self.s.value ** 2)
    >>> 
    >>> # Create and use model
    >>> model = SimpleLinear(input_size=10, output_size=5)
    >>> import numpy as np
    >>> x = np.random.randn(10)
    >>> output = model.predict(x)
    >>> print(output.shape)
    (5,)

    Using with HebbianTrainer:

    >>> from canns.models.brain_inspired import LinearLayer
    >>> from canns.trainer import HebbianTrainer
    >>> 
    >>> # Create model
    >>> model = LinearLayer(input_size=20, output_size=10)
    >>> 
    >>> # Create trainer
    >>> trainer = HebbianTrainer(model, learning_rate=0.01)
    >>> 
    >>> # Generate training patterns
    >>> patterns = [np.random.randn(20) for _ in range(100)]
    >>> 
    >>> # Train
    >>> trainer.train(patterns)
    >>> 
    >>> # Predict
    >>> test_pattern = np.random.randn(20)
    >>> prediction = trainer.predict(test_pattern)

    See Also
    --------
    LinearLayer : Linear model with BCM threshold support
    SpikingLayer : Spiking neural network layer with STDP traces
    AmariHopfieldNetwork : Hopfield network with continuous dynamics
    HebbianTrainer : Generic Hebbian learning trainer
    OjaTrainer : Oja's rule for PCA extraction
    BCMTrainer : BCM rule with sliding threshold
    STDPTrainer : Spike-timing-dependent plasticity

    References
    ----------
    .. [1] Hebb, D. O. (1949). The organization of behavior.
    .. [2] Oja, E. (1982). Simplified neuron model as a principal component analyzer.
    .. [3] Bienenstock, Cooper, & Munro (1982). Theory for the development of neuron selectivity.
    """

    # Default attribute name for Hebbian-compatible weight parameter.
    # Models can override if they expose a differently named matrix.
    @property
    def weight_attr(self) -> str:
        """
        Name of the connection weight attribute used by generic training.

        Override in subclasses if the weight parameter is not named ``W``.

        Returns
        -------
        str
            Name of weight attribute (default: "W")

        Example
        -------
        >>> from canns.models.brain_inspired import LinearLayer
        >>> model = LinearLayer(input_size=10, output_size=5)
        >>> print(model.weight_attr)
        'W'
        >>> # Access weights
        >>> weights = getattr(model, model.weight_attr)
        >>> print(weights.value.shape)
        (5, 10)
        """
        return "W"

    @property
    def predict_state_attr(self) -> str:
        """
        Name of the state vector attribute used by generic prediction.

        Override in subclasses if the prediction state is not stored in ``s``.

        Returns
        -------
        str
            Name of state attribute (default: "s")

        Example
        -------
        >>> from canns.models.brain_inspired import LinearLayer
        >>> model = LinearLayer(input_size=10, output_size=5)
        >>> print(model.predict_state_attr)
        's'
        >>> # Access state
        >>> state = getattr(model, model.predict_state_attr)
        >>> print(state.value.shape)
        (5,)
        """
        return "s"

    def apply_hebbian_learning(self, train_data):
        """
        Optional model-specific Hebbian learning implementation.

        The generic ``HebbianTrainer`` can update ``W`` directly without requiring this
        method. Only implement when custom behavior deviates from the generic rule.

        Parameters
        ----------
        train_data : iterable
            Training patterns to learn

        Raises
        ------
        NotImplementedError
            If called without subclass implementation

        Notes
        -----
        Most models should rely on generic trainer implementation rather than
        overriding this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement `apply_hebbian_learning`"
        )

    def predict(self, pattern):
        """
        Generate prediction for input pattern.

        Must be implemented by subclasses to define model-specific prediction logic.

        Parameters
        ----------
        pattern : array-like
            Input pattern

        Returns
        -------
        array-like
            Model prediction/output

        Raises
        ------
        NotImplementedError
            If called without subclass implementation
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not implement `predict`")

    @property
    def energy(self) -> float:
        """
        Current energy of the model state (used for convergence checks in prediction).

        Implementations may return a float or a 0-dim array; the trainer treats it as a scalar.

        Returns
        -------
        float
            Energy value (lower is typically more stable)

        Raises
        ------
        NotImplementedError
            If called without subclass implementation

        Example
        -------
        >>> from canns.models.brain_inspired import AmariHopfieldNetwork
        >>> model = AmariHopfieldNetwork(num_neurons=100)
        >>> # After some updates...
        >>> energy = model.energy
        >>> print(f"Current energy: {energy:.4f}")
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not implement `energy`")

    def resize(
        self, num_neurons: int, preserve_submatrix: bool = True
    ):  # pragma: no cover - optional
        """
        Optional method to resize model state/parameters to ``num_neurons``.

        Default implementation is a stub. Subclasses may override to support dynamic
        dimensionality changes.

        Parameters
        ----------
        num_neurons : int
            New number of neurons
        preserve_submatrix : bool, default=True
            If True, preserve existing weight values in top-left submatrix

        Raises
        ------
        NotImplementedError
            If called without subclass implementation

        Example
        -------
        >>> from canns.models.brain_inspired import AmariHopfieldNetwork
        >>> model = AmariHopfieldNetwork(num_neurons=50)
        >>> # Resize to accommodate larger patterns
        >>> model.resize(num_neurons=100, preserve_submatrix=True)
        >>> print(model.W.value.shape)
        (100, 100)
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not implement `resize`")


class BrainInspiredModelGroup(BasicModelGroup):
    """
    Base class for groups of brain-inspired models.

    This class manages collections of brain-inspired models and provides
    coordinated learning and dynamics across multiple model instances. It
    extends BasicModelGroup to handle ensembles or hierarchies of brain-inspired
    models with synchronized training and inference.

    Use this when you need to:
    - Manage multiple related models as a cohesive unit
    - Coordinate learning across model hierarchy
    - Share parameters or states between models
    - Implement multi-scale or modular architectures

    Parameters
    ----------
    **kwargs
        Keyword arguments passed to parent BasicModelGroup

    Example
    -------
    Creating a model group:

    >>> from canns.models.brain_inspired import BrainInspiredModelGroup, LinearLayer
    >>> import brainpy.math as bm
    >>> 
    >>> class TwoLayerNetwork(BrainInspiredModelGroup):
    ...     def __init__(self, input_size, hidden_size, output_size):
    ...         super().__init__()
    ...         self.layer1 = LinearLayer(input_size, hidden_size)
    ...         self.layer2 = LinearLayer(hidden_size, output_size)
    ...     
    ...     def forward(self, x):
    ...         h = self.layer1.forward(x)
    ...         y = self.layer2.forward(h)
    ...         return y
    >>> 
    >>> # Create hierarchical network
    >>> network = TwoLayerNetwork(
    ...     input_size=20,
    ...     hidden_size=10,
    ...     output_size=5
    ... )
    >>> 
    >>> # Use the network
    >>> import numpy as np
    >>> x = np.random.randn(20)
    >>> output = network.forward(x)
    >>> print(output.shape)
    (5,)

    See Also
    --------
    BrainInspiredModel : Base class for individual models
    BasicModelGroup : Parent class for generic model groups
    HierarchicalNetwork : Specific implementation of hierarchical path integration

    Notes
    -----
    - Subclasses should register component models as attributes
    - The group can implement collective training and prediction methods
    - Useful for multi-area brain models or deep architectures
    """

    pass
