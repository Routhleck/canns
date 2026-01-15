"""BCM (Bienenstock-Cooper-Munro) sliding-threshold plasticity trainer."""

from __future__ import annotations

from collections.abc import Iterable

import brainpy.math as bm
import jax.numpy as jnp

from ..models.brain_inspired import BrainInspiredModel
from ._base import Trainer

__all__ = ["BCMTrainer"]


class BCMTrainer(Trainer):
    """
    BCM (Bienenstock-Cooper-Munro) sliding-threshold plasticity trainer.

    The BCM rule uses a dynamic postsynaptic threshold to switch between
    potentiation and depression based on recent activity, yielding stable
    receptive-field development and experience-dependent refinement.

    Learning Rule
    -------------
    .. math::

        \\Delta W_{ij} = \\eta y_i (y_i - \\theta_i) x_j

    where:
        - :math:`W_{ij}` is the weight from input j to neuron i
        - :math:`x_j` is the presynaptic activity
        - :math:`y_i` is the postsynaptic activity
        - :math:`\\theta_i` is the modification threshold for neuron i

    The threshold θ evolves as a sliding average:
        θ_i = <y_i^2>

    This creates two regimes:
        - If y > θ: potentiation (LTP, strengthen synapses)
        - If y < θ: depression (LTD, weaken synapses)

    Parameters
    ----------
    model : BrainInspiredModel
        The model to train (typically LinearLayer with use_bcm_threshold=True)
    learning_rate : float, default=0.01
        Learning rate η for weight updates
    weight_attr : str, default="W"
        Name of model attribute holding the connection weights
    compiled : bool, default=True
        Whether to use JIT-compiled training loop
    **kwargs
        Additional arguments passed to parent Trainer

    Example
    -------
    Train with BCM rule for selective receptive fields:

    >>> import numpy as np
    >>> from canns.models.brain_inspired import LinearLayer
    >>> from canns.trainer import BCMTrainer
    >>> 
    >>> # Create model with BCM threshold
    >>> model = LinearLayer(
    ...     input_size=20,
    ...     output_size=5,
    ...     use_bcm_threshold=True,
    ...     threshold_tau=100.0
    ... )
    >>> 
    >>> # Create BCM trainer
    >>> trainer = BCMTrainer(
    ...     model=model,
    ...     learning_rate=0.001
    ... )
    >>> 
    >>> # Generate structured training patterns
    >>> np.random.seed(42)
    >>> # Two types of patterns
    >>> patterns_A = [np.concatenate([np.random.randn(10) * 2, np.zeros(10)])
    ...               for _ in range(50)]
    >>> patterns_B = [np.concatenate([np.zeros(10), np.random.randn(10) * 2])
    ...               for _ in range(50)]
    >>> patterns = patterns_A + patterns_B
    >>> np.random.shuffle(patterns)
    >>> 
    >>> # Train with BCM rule
    >>> trainer.train(patterns)
    >>> 
    >>> # Check learned selectivity
    >>> test_A = np.concatenate([np.ones(10), np.zeros(10)])
    >>> test_B = np.concatenate([np.zeros(10), np.ones(10)])
    >>> response_A = trainer.predict(test_A)
    >>> response_B = trainer.predict(test_B)
    >>> print(f"Response to pattern A: {response_A[:2]}")
    >>> print(f"Response to pattern B: {response_B[:2]}")

    See Also
    --------
    OjaTrainer : Oja's rule without sliding threshold
    HebbianTrainer : Standard Hebbian learning
    LinearLayer : Compatible model with BCM threshold support

    References
    ----------
    .. [1] Bienenstock, E. L., Cooper, L. N., & Munro, P. W. (1982).
           Theory for the development of neuron selectivity: orientation specificity
           and binocular interaction in visual cortex. Journal of Neuroscience, 2(1), 32-48.
    .. [2] Cooper, L. N., & Bear, M. F. (2012). The BCM theory of synapse modification
           at 30: interaction of theory with experiment. Nature Reviews Neuroscience, 13(11), 798-810.

    Notes
    -----
    - Model must have 'theta' attribute for sliding threshold
    - Threshold adapts based on recent activity history
    - Enables stable, selective receptive field development
    - Weight clipping prevents divergence
    """

    def __init__(
        self,
        model: BrainInspiredModel,
        learning_rate: float = 0.01,
        weight_attr: str = "W",
        compiled: bool = True,
        **kwargs,
    ):
        """
        Initialize BCM trainer.

        Args:
            model: The model to train (typically LinearLayer with use_bcm_threshold=True)
            learning_rate: Learning rate η for weight updates
            weight_attr: Name of model attribute holding the connection weights
            compiled: Whether to use JIT-compiled training loop (default: True)
            **kwargs: Additional arguments passed to parent Trainer
        """
        super().__init__(model=model, **kwargs)
        self.learning_rate = learning_rate
        self.weight_attr = weight_attr
        self.compiled = compiled

    def train(self, train_data: Iterable):
        """
        Train the model using BCM rule.

        Args:
            train_data: Iterable of input patterns (each of shape (input_size,))
        """
        # Get weight parameter
        weight_param = getattr(self.model, self.weight_attr, None)
        if weight_param is None or not hasattr(weight_param, "value"):
            raise AttributeError(
                f"Model does not have a '{self.weight_attr}' parameter with .value attribute"
            )

        # Check if model has theta (sliding threshold)
        if not hasattr(self.model, "theta"):
            raise AttributeError("Model must have 'theta' attribute for BCM learning")

        if self.compiled:
            self._train_compiled(train_data, weight_param)
        else:
            self._train_uncompiled(train_data, weight_param)

    def _train_compiled(self, train_data: Iterable, weight_param):
        """
        JIT-compiled training loop using bp.transform.scan.

        Args:
            train_data: Iterable of input patterns
            weight_param: Weight parameter object
        """
        # Convert patterns to array for JIT compilation
        patterns = jnp.stack([jnp.asarray(p, dtype=jnp.float32) for p in train_data])

        # Get threshold tau from model
        threshold_tau = getattr(self.model, "threshold_tau", 100.0)

        # Initial state
        W_init = jnp.asarray(weight_param.value, dtype=jnp.float32)
        theta_init = jnp.asarray(self.model.theta.value, dtype=jnp.float32)

        # Training step for single pattern
        def train_step(carry, x):
            W, theta = carry

            # Forward pass
            y = W @ x

            # BCM rule: ΔW = η * y * (y - θ) * x^T
            phi = y * (y - theta)
            delta_W = self.learning_rate * jnp.outer(phi, x)
            W = W + delta_W

            # Clip weights
            W = jnp.clip(W, -10.0, 10.0)

            # Update threshold: θ ← θ + (1/τ) * (y² - θ)
            y_squared = y**2
            alpha = 1.0 / threshold_tau if threshold_tau > 0 else 1.0
            theta = theta + alpha * (y_squared - theta)

            return (W, theta), None

        # Run compiled scan
        (W_final, theta_final), _ = bm.scan(train_step, (W_init, theta_init), patterns)

        # Update model parameters
        weight_param.value = W_final
        self.model.theta.value = theta_final

    def _train_uncompiled(self, train_data: Iterable, weight_param):
        """
        Python loop training (fallback, slower but more flexible).

        Args:
            train_data: Iterable of input patterns
            weight_param: Weight parameter object
        """
        W = weight_param.value

        # Process each pattern
        for pattern in train_data:
            x = jnp.asarray(pattern, dtype=jnp.float32)

            # Forward pass: y = W @ x
            if hasattr(self.model, "forward"):
                y = self.model.forward(x)
            else:
                y = W @ x

            # Get current threshold
            theta = self.model.theta.value

            # BCM rule: ΔW = η * y * (y - θ) * x^T
            phi = y * (y - theta)  # BCM modulation function
            delta_W = self.learning_rate * jnp.outer(phi, x)

            W = W + delta_W

            # Clip weights to prevent divergence
            W = jnp.clip(W, -10.0, 10.0)

            # Update threshold using model's method
            if hasattr(self.model, "update_threshold"):
                self.model.update_threshold()

        # Update model weights
        weight_param.value = W

    def predict(self, pattern, *args, **kwargs):
        """
        Predict output for a single input pattern.

        Args:
            pattern: Input pattern of shape (input_size,)

        Returns:
            Output pattern of shape (output_size,)
        """
        if hasattr(self.model, "forward"):
            return self.model.forward(pattern)
        else:
            # Fallback: direct computation
            weight_param = getattr(self.model, self.weight_attr)
            x = jnp.asarray(pattern, dtype=jnp.float32)
            return weight_param.value @ x
