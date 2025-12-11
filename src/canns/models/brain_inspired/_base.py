from ..basic._base import BasicModel, BasicModelGroup


class BrainInspiredModel(BasicModel):
    """
    Base class for brain-inspired models.

    Trainer compatibility notes
    - If a model wants to support generic Hebbian training, expose a weight parameter
      attribute with a ``.value`` array of shape (N, N) (commonly a
      ``bm.Variable``). The recommended attribute name is ``W``.
    - Override ``weight_attr`` to declare a different attribute name if needed. Models
      that use standard backprop may omit this entirely.
    - Implementing ``apply_hebbian_learning`` is optional; prefer letting the trainer
      handle the generic rule when applicable. Implement this only when you need
      model-specific behavior.

    Notes on Predict compatibility
    - For the trainer's generic prediction path, models typically expose:
      1) an ``update(prev_energy)`` method to advance one step (optional; not all models
         require energy-driven updates),
      2) an ``energy`` property to compute current energy (scalar-like),
      3) a state vector attribute (default ``s``) with ``.value`` as 1D array used as
         the prediction state; override ``predict_state_attr`` to change the name.

    Optional resizing
    - Models may implement ``resize(num_neurons: int, preserve_submatrix: bool = True)`` to
      allow trainers to change neuron dimensionality on the fly (e.g., when training with
      patterns of a different length). When implemented, the trainer will call this to
      align dimensions before training/prediction.
    """

    # Default attribute name for Hebbian-compatible weight parameter.
    # Models can override if they expose a differently named matrix.
    @property
    def weight_attr(self) -> str:
        """
        Name of the connection weight attribute used by generic training.

        Override in subclasses if the weight parameter is not named ``W``.
        """
        return "W"

    @property
    def predict_state_attr(self) -> str:
        """
        Name of the state vector attribute used by generic prediction.

        Override in subclasses if the prediction state is not stored in ``s``.
        """
        return "s"

    def apply_hebbian_learning(self, train_data):
        """
        Optional model-specific Hebbian learning implementation.

        The generic ``HebbianTrainer`` can update ``W`` directly without requiring this
        method. Only implement when custom behavior deviates from the generic rule.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement `apply_hebbian_learning`"
        )

    def predict(self, pattern):
        raise NotImplementedError(f"{self.__class__.__name__} does not implement `predict`")

    @property
    def energy(self) -> float:
        """
        Current energy of the model state (used for convergence checks in prediction).

        Implementations may return a float or a 0-dim array; the trainer treats it as a scalar.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not implement `energy`")

    def resize(
        self, num_neurons: int, preserve_submatrix: bool = True
    ):  # pragma: no cover - optional
        """
        Optional method to resize model state/parameters to ``num_neurons``.

        Default implementation is a stub. Subclasses may override to support dynamic
        dimensionality changes.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not implement `resize`")


class BrainInspiredModelGroup(BasicModelGroup):
    """
    Base class for groups of brain-inspired models.

    This class manages collections of brain-inspired models and provides
    coordinated learning and dynamics across multiple model instances.
    """

    pass
