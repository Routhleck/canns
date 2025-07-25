from typing import Union, Tuple, Sequence
import numpy as np
from brainunit import Quantity

# External input types
Iext_type = float | Quantity
Iext_pair_type = Iext_type | tuple[Iext_type, ...]

# Time-related types
time_type = float | Quantity

# Position types for stimuli
Position1D = float | Quantity
Position2D = Tuple[float, float] | Tuple[Quantity, Quantity]
PositionND = Union[Position1D, Position2D, Sequence[float], Sequence[Quantity]]

# Neural network shape types
NetworkShape = Union[int, Tuple[int, ...]]

# Array-like types for JAX/NumPy compatibility
ArrayLike = Union[np.ndarray, Sequence[float]]
