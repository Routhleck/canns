"""Checkpoint utilities for saving and loading trained RNN models using BrainState's built-in checkpointing."""

import os

import brainstate as bst
import braintools as bts

__all__ = ["save_checkpoint", "load_checkpoint"]


def save_checkpoint(model: bst.nn.Module, filepath: str) -> None:
    """Save model parameters to a checkpoint file using BrainState checkpointing.

    Args:
        model: BrainState model to save.
        filepath: Path to save the checkpoint file.

    Example:
        >>> from canns.analyzer.slow_points import save_checkpoint
        >>> save_checkpoint(rnn, "my_model.msgpack")
        Saved checkpoint to: my_model.msgpack
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)

    # Use BrainState's built-in checkpointing
    states = bst.graph.states(model)
    checkpoint = states[0].to_nest() if isinstance(states, tuple) else states.to_nest()
    bts.file.msgpack_save(filepath, checkpoint)
    print(f"Saved checkpoint to: {filepath}")


def load_checkpoint(model: bst.nn.Module, filepath: str) -> bool:
    """Load model parameters from a checkpoint file using BrainState checkpointing.

    Args:
        model: BrainState model to load parameters into.
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
    if not os.path.exists(filepath):
        return False

    # Use BrainState's built-in checkpointing
    states = bst.graph.states(model)
    checkpoint = states[0].to_nest() if isinstance(states, tuple) else states.to_nest()
    bts.file.msgpack_load(filepath, checkpoint)
    print(f"Loaded checkpoint from: {filepath}")
    return True
