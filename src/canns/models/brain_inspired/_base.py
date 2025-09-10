from ..basic._base import BasicModel, BasicModelGroup


class BrainInspiredModel(BasicModel):
    """
    Base class for brain-inspired models.

    This class extends BasicModel to provide common functionality for brain-inspired
    neural network models, including specialized learning mechanisms, plasticity rules,
    and biological constraints.
    """

    pass

    def apply_hebbian_learning(self, train_data):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement `apply_hebbian_learning`"
        )

    def predict(self, pattern):
        raise NotImplementedError(f"{self.__class__.__name__} does not implement `predict`")


class BrainInspiredGroup(BasicModelGroup):
    """
    Base class for groups of brain-inspired models.

    This class manages collections of brain-inspired models and provides
    coordinated learning and dynamics across multiple model instances.
    """

    pass
