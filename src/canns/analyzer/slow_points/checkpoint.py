"""Checkpoint utilities for saving and loading trained RNN models using BrainPy's built-in checkpointing."""

import os

import brainpy as bp

__all__ = ["save_checkpoint", "load_checkpoint"]


def save_checkpoint(model: bp.DynamicalSystem, filepath: str) -> None:
    """Save model parameters to a checkpoint file using BrainPy checkpointing.

    Args:
        model: BrainPy model to save.
        filepath: Path to save the checkpoint file.

    Example:
        >>> from canns.analyzer.slow_points import save_checkpoint
        >>> save_checkpoint(rnn, "my_model.msgpack")
        Saved checkpoint to: my_model.msgpack
    """
    # Extract all states from model (parameters and state variables)
    states = bp.save_state(model)

    # Save to disk using BrainPy's checkpoint system (automatically creates parent directories)
    bp.checkpoints.save_pytree(filepath, states, overwrite=True)

    # Print confirmation message
    print(f"Saved checkpoint to: {filepath}")


def load_checkpoint(model: bp.DynamicalSystem, filepath: str) -> bool:
    """Load model parameters from a checkpoint file using BrainPy checkpointing.

    Args:
        model: BrainPy model to load parameters into.
        filepath: Path to the checkpoint file.

    Returns:
        True if checkpoint was loaded successfully, False otherwise.

    Example:
        >>> from canns.analyzer.slow_points import load_checkpoint
        >>> if load_checkpoint(rnn, "my_model.msgpack"):
        ...     print("Loaded successfully")
        ... else:
        ...     print("No checkpoint found")
        Loaded checkpoint from: my_model.msgpack
        Loaded successfully
    """
    # Check if file exists
    if not os.path.exists(filepath):
        return False

    try:
        # Load state dictionary from disk
        state_dict = bp.checkpoints.load_pytree(filepath)

        # Load state into model
        bp.load_state(model, state_dict)

        # Print confirmation message
        print(f"Loaded checkpoint from: {filepath}")
        return True

    except (ValueError, FileNotFoundError, OSError):
        # Handle file not found, corrupt file, or permission errors
        return False
