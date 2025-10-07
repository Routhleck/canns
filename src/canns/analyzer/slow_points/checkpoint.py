"""Checkpoint utilities for saving and loading trained RNN models."""

import os
import pickle
from typing import Any

import jax.numpy as jnp
import numpy as np
import brainstate as bst

__all__ = ["save_checkpoint", "load_checkpoint"]


def save_checkpoint(model: bst.nn.Module, filepath: str) -> None:
    """Save model parameters to a checkpoint file.

    Args:
        model: BrainState model to save.
        filepath: Path to save the checkpoint file.

    Example:
        >>> from canns.analyzer.slow_points import save_checkpoint
        >>> save_checkpoint(rnn, "my_model.pkl")
        Saved checkpoint to: my_model.pkl
    """
    # Extract all parameter states
    params = {}
    for name_tuple, state in model.states().items():
        if isinstance(state, bst.ParamState):
            # Convert tuple key to string for easier handling
            if isinstance(name_tuple, tuple):
                name = name_tuple[0]  # Extract string from tuple
            else:
                name = name_tuple
            params[name] = np.array(state.value)

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)

    with open(filepath, "wb") as f:
        pickle.dump(params, f)
    print(f"Saved checkpoint to: {filepath}")


def load_checkpoint(model: bst.nn.Module, filepath: str) -> bool:
    """Load model parameters from a checkpoint file.

    Args:
        model: BrainState model to load parameters into.
        filepath: Path to the checkpoint file.

    Returns:
        True if checkpoint was loaded successfully, False otherwise.

    Example:
        >>> from canns.analyzer.slow_points import load_checkpoint
        >>> if load_checkpoint(rnn, "my_model.pkl"):
        ...     print("Loaded successfully")
        ... else:
        ...     print("No checkpoint found")
        Loaded checkpoint from: my_model.pkl
        Loaded successfully
    """
    if not os.path.exists(filepath):
        return False

    with open(filepath, "rb") as f:
        params = pickle.load(f)

    # Load parameters into model - match with states dictionary
    states_dict = model.states()
    for name_tuple, state in states_dict.items():
        if isinstance(state, bst.ParamState):
            # Extract string name from tuple
            if isinstance(name_tuple, tuple):
                name = name_tuple[0]
            else:
                name = name_tuple

            if name in params:
                state.value = jnp.array(params[name])

    print(f"Loaded checkpoint from: {filepath}")
    return True
