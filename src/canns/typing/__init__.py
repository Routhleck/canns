from collections.abc import Sequence

from brainunit import Quantity

Iext_type = float | Quantity
Iext_pair_type = Sequence[Iext_type]  # For 2D coordinates like [x, y] or (x, y)

time_type = float | Quantity
