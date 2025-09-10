import jax.numpy as jnp

from ..models import BrainInspiredModel
from ..models.brain_inspired.hopfield import AmariHopfieldNetwork, DiscreteHopfieldNetwork

__all__ = ["HebbianTrainer"]


class HebbianTrainer:
    """
    Simple Hebbian trainer for Hopfield networks.

    This trainer implements basic Hebbian learning following BrainState's
    simple training approach for associative memory networks.
    """

    def __init__(
        self,
        model: BrainInspiredModel,
    ):
        """
        Initialize Hebbian trainer.

        Args:
            model (BrainInspiredModel): The model to train.
        """
        self.model = model

    def train(self, train_data):
        self.model.apply_hebbian_learning(train_data)

    def predict(self, pattern):
        return self.model.predict(pattern)