"""STDP (Spike-Timing-Dependent Plasticity) trainer."""

from __future__ import annotations

from collections.abc import Iterable

import brainpy.math as bm
import jax.numpy as jnp

from ..models.brain_inspired import BrainInspiredModel
from ._base import Trainer

__all__ = ["STDPTrainer"]


class STDPTrainer(Trainer):
    """
    STDP (Spike-Timing-Dependent Plasticity) trainer.

    STDP is a biologically-inspired learning rule that adjusts synaptic weights
    based on the precise timing of pre- and post-synaptic spikes. Synapses are
    strengthened when pre-synaptic spikes precede post-synaptic spikes (LTP),
    and weakened when the order is reversed (LTD).

    Trace-based Learning Rule
    --------------------------
    .. math::

        \\Delta W_{ij} = A_{+} \\cdot \\text{trace}_{\\text{pre}}[j] \\cdot \\text{spike}_{\\text{post}}[i] 
                        - A_{-} \\cdot \\text{trace}_{\\text{post}}[i] \\cdot \\text{spike}_{\\text{pre}}[j]

    where:
        - :math:`W_{ij}` is the weight from input j to neuron i
        - spike_pre[j], spike_post[i] are binary spike indicators (0 or 1)
        - trace_pre[j], trace_post[i] are exponential traces of spikes
        - A_plus controls LTP (long-term potentiation) magnitude
        - A_minus controls LTD (long-term depression) magnitude

    The spike traces evolve as:
        trace = decay * trace + spike

    This provides a temporal window for spike-timing correlations.

    Parameters
    ----------
    model : BrainInspiredModel
        The spiking model to train (typically SpikingLayer)
    learning_rate : float, default=0.01
        Global learning rate multiplier
    A_plus : float, default=0.005
        LTP magnitude (potentiation when pre before post)
    A_minus : float, default=0.00525
        LTD magnitude (depression when post before pre), slightly > A_plus for stability
    weight_attr : str, default="W"
        Name of model attribute holding the connection weights
    w_min : float, default=0.0
        Minimum weight value (typically 0 for excitatory synapses)
    w_max : float, default=1.0
        Maximum weight value
    compiled : bool, default=True
        Whether to use JIT-compiled training loop
    **kwargs
        Additional arguments passed to parent Trainer

    Example
    -------
    Train spiking network with STDP:

    >>> import numpy as np
    >>> from canns.models.brain_inspired import SpikingLayer
    >>> from canns.trainer import STDPTrainer
    >>> 
    >>> # Create spiking model
    >>> model = SpikingLayer(
    ...     input_size=20,
    ...     output_size=10,
    ...     threshold=1.0,
    ...     trace_decay=0.95
    ... )
    >>> 
    >>> # Create STDP trainer
    >>> trainer = STDPTrainer(
    ...     model=model,
    ...     learning_rate=0.01,
    ...     A_plus=0.005,
    ...     A_minus=0.00525,
    ...     w_min=0.0,
    ...     w_max=1.0
    ... )
    >>> 
    >>> # Generate spike patterns (Poisson-like)
    >>> np.random.seed(42)
    >>> n_patterns = 100
    >>> spike_patterns = [
    ...     (np.random.rand(20) < 0.1).astype(float)  # 10% spike probability
    ...     for _ in range(n_patterns)
    ... ]
    >>> 
    >>> # Train with STDP
    >>> trainer.train(spike_patterns)
    >>> 
    >>> # Test prediction
    >>> test_spikes = (np.random.rand(20) < 0.15).astype(float)
    >>> output_spikes = trainer.predict(test_spikes)
    >>> print(f"Input spikes: {test_spikes.sum()}")
    >>> print(f"Output spikes: {output_spikes.sum()}")
    >>> print(f"Output spike pattern: {output_spikes}")

    Temporal pattern learning:

    >>> # Learn specific spike timing patterns
    >>> # Pattern: spike at position 0, then 5, then 10
    >>> temporal_pattern = np.zeros(20)
    >>> temporal_pattern[[0, 5, 10]] = 1.0
    >>> 
    >>> # Repeat pattern to strengthen learning
    >>> patterns = [temporal_pattern for _ in range(50)]
    >>> 
    >>> trainer_temporal = STDPTrainer(
    ...     model=SpikingLayer(20, 5),
    ...     A_plus=0.01,
    ...     A_minus=0.01
    ... )
    >>> trainer_temporal.train(patterns)
    >>> 
    >>> # Test with learned pattern
    >>> response = trainer_temporal.predict(temporal_pattern)
    >>> print(f"Response to learned pattern: {response.sum()} spikes")

    See Also
    --------
    SpikingLayer : Compatible spiking neural network model
    HebbianTrainer : Rate-based Hebbian learning
    BCMTrainer : BCM rule with threshold

    References
    ----------
    .. [1] Bi, G. Q., & Poo, M. M. (1998). Synaptic modifications in cultured
           hippocampal neurons: dependence on spike timing, synaptic strength, and
           postsynaptic cell type. Journal of neuroscience, 18(24), 10464-10472.
    .. [2] Song, S., Miller, K. D., & Abbott, L. F. (2000). Competitive Hebbian learning
           through spike-timing-dependent synaptic plasticity. Nature neuroscience, 3(9), 919-926.
    .. [3] Morrison, A., Diesmann, M., & Gerstner, W. (2008). Phenomenological models
           of synaptic plasticity based on spike timing. Biological cybernetics, 98(6), 459-478.

    Notes
    -----
    - Model must have 'trace_pre' and 'trace_post' attributes
    - A_minus is typically slightly larger than A_plus for stability
    - Weights are clipped to [w_min, w_max] to prevent divergence
    - Trace decay controls the temporal window of spike correlations
    - Common in spiking neural network (SNN) training
    """

    def __init__(
        self,
        model: BrainInspiredModel,
        learning_rate: float = 0.01,
        A_plus: float = 0.005,
        A_minus: float = 0.00525,
        weight_attr: str = "W",
        w_min: float = 0.0,
        w_max: float = 1.0,
        compiled: bool = True,
        **kwargs,
    ):
        """
        Initialize STDP trainer.

        Args:
            model: The spiking model to train (typically SpikingLayer)
            learning_rate: Global learning rate multiplier (default: 0.01)
            A_plus: LTP magnitude (default: 0.005)
            A_minus: LTD magnitude (default: 0.00525, slightly > A_plus for stability)
            weight_attr: Name of model attribute holding the connection weights
            w_min: Minimum weight value (default: 0.0 for excitatory synapses)
            w_max: Maximum weight value (default: 1.0)
            compiled: Whether to use JIT-compiled training loop (default: True)
            **kwargs: Additional arguments passed to parent Trainer
        """
        super().__init__(model=model, **kwargs)
        self.learning_rate = learning_rate
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.weight_attr = weight_attr
        self.w_min = w_min
        self.w_max = w_max
        self.compiled = compiled

    def train(self, train_data: Iterable):
        """
        Train the model using STDP rule.

        Args:
            train_data: Iterable of input spike patterns (each of shape (input_size,))
                       Each pattern should contain binary values (0 or 1)
        """
        # Get weight parameter
        weight_param = getattr(self.model, self.weight_attr, None)
        if weight_param is None or not hasattr(weight_param, "value"):
            raise AttributeError(
                f"Model does not have a '{self.weight_attr}' parameter with .value attribute"
            )

        # Check if model has required trace attributes
        if not hasattr(self.model, "trace_pre"):
            raise AttributeError("Model must have 'trace_pre' attribute for STDP learning")
        if not hasattr(self.model, "trace_post"):
            raise AttributeError("Model must have 'trace_post' attribute for STDP learning")

        if self.compiled:
            self._train_compiled(train_data, weight_param)
        else:
            self._train_uncompiled(train_data, weight_param)

    def _train_compiled(self, train_data: Iterable, weight_param):
        """
        JIT-compiled training loop using bp.transform.scan.

        Args:
            train_data: Iterable of input spike patterns
            weight_param: Weight parameter object
        """
        # Convert patterns to array for JIT compilation
        patterns = jnp.stack([jnp.asarray(p, dtype=jnp.float32) for p in train_data])

        # Get model parameters
        trace_decay = getattr(self.model, "trace_decay", 0.95)
        threshold = getattr(self.model, "threshold", 1.0)
        v_reset = getattr(self.model, "v_reset", 0.0)
        leak = getattr(self.model, "leak", 0.9)

        # Initial state
        W_init = jnp.asarray(weight_param.value, dtype=jnp.float32)
        trace_pre_init = jnp.zeros(self.model.input_size, dtype=jnp.float32)
        trace_post_init = jnp.zeros(self.model.output_size, dtype=jnp.float32)
        v_init = jnp.zeros(self.model.output_size, dtype=jnp.float32)

        # Training step for single pattern
        def train_step(carry, x):
            W, trace_pre, trace_post, v = carry

            # Update pre-synaptic trace
            trace_pre = trace_decay * trace_pre + x

            # Forward pass (LIF dynamics)
            input_current = W @ x
            v = leak * v + input_current

            # Generate spikes
            spike_post = (v >= threshold).astype(jnp.float32)

            # Reset membrane potential
            v = jnp.where(spike_post > 0, v_reset, v)

            # Update post-synaptic trace
            trace_post = trace_decay * trace_post + spike_post

            # STDP weight update
            # LTP: pre before post (pre trace high, post spike now)
            ltp = self.A_plus * jnp.outer(spike_post, trace_pre)
            # LTD: post before pre (post trace high, pre spike now)
            ltd = self.A_minus * jnp.outer(trace_post, x)

            delta_W = self.learning_rate * (ltp - ltd)
            W = W + delta_W

            # Clip weights to valid range
            W = jnp.clip(W, self.w_min, self.w_max)

            return (W, trace_pre, trace_post, v), None

        # Run compiled scan
        (W_final, trace_pre_final, trace_post_final, v_final), _ = bm.scan(
            train_step, (W_init, trace_pre_init, trace_post_init, v_init), patterns
        )

        # Update model parameters
        weight_param.value = W_final
        self.model.trace_pre.value = trace_pre_final
        self.model.trace_post.value = trace_post_final
        self.model.v.value = v_final

    def _train_uncompiled(self, train_data: Iterable, weight_param):
        """
        Python loop training (fallback, slower but more flexible).

        Args:
            train_data: Iterable of input spike patterns
            weight_param: Weight parameter object
        """
        W = weight_param.value

        # Process each pattern
        for pattern in train_data:
            x = jnp.asarray(pattern, dtype=jnp.float32)

            # Store current traces before forward pass
            trace_pre_before = self.model.trace_pre.value
            trace_post_before = self.model.trace_post.value

            # Forward pass through model (updates traces and generates spikes)
            spike_post = self.model.forward(x)

            # STDP weight update
            # LTP: pre before post (use pre trace from before post spike)
            ltp = self.A_plus * jnp.outer(spike_post, trace_pre_before)
            # LTD: post before pre (use post trace from before pre spike)
            ltd = self.A_minus * jnp.outer(trace_post_before, x)

            delta_W = self.learning_rate * (ltp - ltd)
            W = W + delta_W

            # Clip weights to valid range
            W = jnp.clip(W, self.w_min, self.w_max)

        # Update model weights
        weight_param.value = W

    def predict(self, pattern, *args, **kwargs):
        """
        Predict output spikes for a single input spike pattern.

        Args:
            pattern: Input spike pattern of shape (input_size,)

        Returns:
            Output spike pattern of shape (output_size,) with binary values (0 or 1)
        """
        if hasattr(self.model, "forward"):
            return self.model.forward(pattern)
        else:
            # Fallback: direct computation with thresholding
            weight_param = getattr(self.model, self.weight_attr)
            x = jnp.asarray(pattern, dtype=jnp.float32)
            v = weight_param.value @ x
            threshold = getattr(self.model, "threshold", 1.0)
            return (v >= threshold).astype(jnp.float32)
