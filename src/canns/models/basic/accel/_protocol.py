"""Backend protocol for the CANN recurrent matvec ``Irec = W @ r``.

This module only contains the :class:`IrecBackend` Protocol used as
a structural type for the strategy objects held on
``model.irec_backend``. Concrete implementations live in
:mod:`.backends`; the dispatcher that selects one lives in
:mod:`.factory`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import jax


@runtime_checkable
class IrecBackend(Protocol):
    """Callable strategy: ``backend(r) -> Irec``.

    Concrete backends must carry:

    - ``mode`` (str): the public ``accl_mode`` value
    - ``k``   (int): the effective rank (``-1`` for full-rank / exact)
    - ``U``, ``V``, ``K_fft``: the pre-computed factors (any of them
      may be ``None`` for backends that do not use them). The model
      exposes these via properties for backward compatibility.
    """

    mode: str
    k: int

    def __call__(self, r: jax.Array) -> jax.Array: ...
