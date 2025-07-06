from brainunit import Quantity

Iext_type = float | Quantity
Iext_pair_type = Iext_type | tuple[Iext_type, ...]

time_type = float | Quantity
