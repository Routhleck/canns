"""Oja's normalized Hebbian learning trainer."""

from __future__ import annotations

from collections.abc import Iterable

import brainpy.math as bm
import jax.numpy as jnp

from ..models.brain_inspired import BrainInspiredModel
from ._base import Trainer
from .utils import normalize_weight_rows

__all__ = ["OjaTrainer"]


class OjaTrainer(Trainer):
    """
    Oja's normalized Hebbian learning trainer.

    Oja's rule stabilizes pure Hebbian growth by introducing a weight-dependent
    normalization term, enabling single-neuron principal component extraction
    without unbounded weight magnitudes.

    Learning Rule
    -------------
    .. math::

        \\Delta W_{ij} = \\eta (y_i x_j - y_i^2 W_{ij})

    where:
        - :math:`W_{ij}` is the weight from input j to output i
        - :math:`x_j` is the input activity
        - :math:`y_i` is the output activity (y = W @ x)
        - :math:`\\eta` is the learning rate

    The rule can be rewritten as:
        ΔW = η * (y @ x^T - diag(y^2) @ W)

    This naturally leads to weight normalization and PCA extraction.

    Parameters
    ----------
    model : BrainInspiredModel
        The model to train (typically LinearLayer)
    learning_rate : float, default=0.01
        Learning rate η for weight updates
    normalize_weights : bool, default=True
        Whether to normalize weights to unit norm after each update
    weight_attr : str, default="W"
        Name of model attribute holding the connection weights
    compiled : bool, default=True
        Whether to use JIT-compiled training loop for efficiency
    **kwargs
        Additional arguments passed to parent Trainer

    Attributes
    ----------
    learning_rate : float
        Stored learning rate
    normalize_weights : bool
        Weight normalization flag
    weight_attr : str
        Name of weight parameter
    compiled : bool
        Compilation flag

    Methods
    -------
    train(train_data)
        Train model using Oja's rule
    predict(pattern, *args, **kwargs)
        Generate prediction for input pattern

    Example
    -------
    Extract principal component from data:

    >>> import numpy as np
    >>> from canns.models.brain_inspired import LinearLayer
    >>> from canns.trainer import OjaTrainer
    >>> 
    >>> # Create model (1 output neuron extracts 1st PC)
    >>> model = LinearLayer(input_size=10, output_size=1)
    >>> 
    >>> # Create trainer
    >>> trainer = OjaTrainer(
    ...     model=model,
    ...     learning_rate=0.01,
    ...     normalize_weights=True
    ... )
    >>> 
    >>> # Generate correlated training data
    >>> np.random.seed(42)
    >>> # Data mostly along first axis
    >>> patterns = [
    ...     np.random.randn(10) * [2.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    ...     for _ in range(100)
    ... ]
    >>> 
    >>> # Train to extract principal component
    >>> trainer.train(patterns)
    >>> 
    >>> # Check learned weights (should be largest for first dimension)
    >>> weights = model.W.value
    >>> print(f"Weight magnitudes: {np.abs(weights[0])[:3]}")  # First 3 dims
    >>> 
    >>> # Test prediction
    >>> test_pattern = np.random.randn(10)
    >>> pc_value = trainer.predict(test_pattern)
    >>> print(f"PC projection: {pc_value}")

    Multiple principal components:

    >>> # Extract 3 principal components
    >>> model_3pc = LinearLayer(input_size=10, output_size=3)
    >>> trainer_3pc = OjaTrainer(model_3pc, learning_rate=0.005)
    >>> 
    >>> # Train
    >>> trainer_3pc.train(patterns)
    >>> 
    >>> # Each output neuron now represents one PC
    >>> pcs = trainer_3pc.predict(test_pattern)
    >>> print(f"3 PC projections shape: {pcs.shape}")

    See Also
    --------
    SangerTrainer : Multiple orthogonal PCs with Gram-Schmidt
    HebbianTrainer : Standard Hebbian learning without normalization
    LinearLayer : Compatible linear model

    References
    ----------
    .. [1] Oja, E. (1982). Simplified neuron model as a principal component analyzer.
           Journal of Mathematical Biology, 15(3), 267-273.
    .. [2] Oja, E. (1989). Neural networks, principal components, and subspaces.
           International journal of neural systems, 1(01), 61-68.

    Notes
    -----
    - The single-neuron version extracts the first principal component
    - Multi-neuron version may extract multiple PCs but they aren't guaranteed orthogonal
    - For orthogonal PCs, use SangerTrainer instead
    - Weight normalization is recommended for stability
    """

    def __init__(
        self,
        model: BrainInspiredModel,
        learning_rate: float = 0.01,
        normalize_weights: bool = True,
        weight_attr: str = "W",
        compiled: bool = True,
        **kwargs,
    ):
        """
        Initialize Oja trainer.

        Args:
            model: The model to train (typically LinearLayer)
            learning_rate: Learning rate η for weight updates
            normalize_weights: Whether to normalize weights to unit norm after each update
            weight_attr: Name of model attribute holding the connection weights
            compiled: Whether to use JIT-compiled training loop (default: True)
            **kwargs: Additional arguments passed to parent Trainer
        """
        super().__init__(model=model, **kwargs)
        self.learning_rate = learning_rate
        self.normalize_weights = normalize_weights
        self.weight_attr = weight_attr
        self.compiled = compiled

    def train(self, train_data: Iterable):
        """
        Train the model using Oja's rule.

        Args:
            train_data: Iterable of input patterns (each of shape (input_size,))
        """
        # Get weight parameter
        weight_param = getattr(self.model, self.weight_attr, None)
        if weight_param is None or not hasattr(weight_param, "value"):
            raise AttributeError(
                f"Model does not have a '{self.weight_attr}' parameter with .value attribute"
            )

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

        # Initial weights
        W_init = jnp.asarray(weight_param.value, dtype=jnp.float32)

        # Training step for single pattern
        def train_step(W, x):
            # Compute output: y = W @ x
            y = W @ x

            # Oja's rule: ΔW = η * (y @ x^T - diag(y^2) @ W)
            outer_product = jnp.outer(y, x)
            normalization = jnp.outer(y * y, jnp.ones_like(x)) * W

            delta_W = self.learning_rate * (outer_product - normalization)
            W = W + delta_W

            # Optional: normalize weights to unit norm
            if self.normalize_weights:
                W = normalize_weight_rows(W)

            return W, None

        # Run compiled scan
        W_final, _ = bm.scan(train_step, W_init, patterns)

        # Update model parameters
        weight_param.value = W_final

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

            # Compute output: y = W @ x
            y = W @ x

            # Oja's rule: ΔW = η * (y @ x^T - diag(y^2) @ W)
            outer_product = jnp.outer(y, x)
            normalization = jnp.outer(y * y, jnp.ones_like(x)) * W

            delta_W = self.learning_rate * (outer_product - normalization)
            W = W + delta_W

            # Optional: normalize weights to unit norm
            if self.normalize_weights:
                W = normalize_weight_rows(W)

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
