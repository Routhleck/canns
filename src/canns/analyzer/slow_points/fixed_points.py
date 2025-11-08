"""FixedPoints data container class for storing fixed point analysis results."""

import numpy as np


class FixedPoints:
    """Container for storing and manipulating fixed points.

    This class stores fixed points found by the FixedPointFinder algorithm,
    along with associated metadata like Jacobians, eigenvalues, and stability.

    Attributes:
        xstar: [n x n_states] array of fixed point states.
        F_xstar: [n x n_states] array of states after one RNN step from xstar.
        x_init: [n x n_states] array of initial states used for optimization.
        inputs: [n x n_inputs] array of constant inputs during optimization.
        qstar: [n,] array of final q values (optimization objective).
        dq: [n,] array of change in q at the last optimization step.
        n_iters: [n,] array of iteration counts for each optimization.
        J_xstar: [n x n_states x n_states] array of Jacobians dF/dx at fixed points.
        dFdu: [n x n_states x n_inputs] array of Jacobians dF/du at fixed points.
        eigval_J_xstar: [n x n_states] complex array of eigenvalues.
        eigvec_J_xstar: [n x n_states x n_states] complex array of eigenvectors.
        is_stable: [n,] bool array indicating stability (max |eigenvalue| < 1).
        cond_id: [n,] array of condition IDs (optional).
        tol_unique: Tolerance for identifying unique fixed points.
        dtype: NumPy dtype for data storage.
    """

    def __init__(
        self,
        xstar: np.ndarray | None = None,
        F_xstar: np.ndarray | None = None,
        x_init: np.ndarray | None = None,
        inputs: np.ndarray | None = None,
        qstar: np.ndarray | None = None,
        dq: np.ndarray | None = None,
        n_iters: np.ndarray | None = None,
        J_xstar: np.ndarray | None = None,
        dFdu: np.ndarray | None = None,
        eigval_J_xstar: np.ndarray | None = None,
        eigvec_J_xstar: np.ndarray | None = None,
        is_stable: np.ndarray | None = None,
        cond_id: np.ndarray | None = None,
        tol_unique: float = 1e-3,
        dtype=np.float32,
    ):
        """Initialize a FixedPoints object.

        Args:
            xstar: Fixed point states [n x n_states].
            F_xstar: States after one RNN step [n x n_states].
            x_init: Initial states [n x n_states].
            inputs: Constant inputs [n x n_inputs].
            qstar: Final q values [n,].
            dq: Change in q at last step [n,].
            n_iters: Iteration counts [n,].
            J_xstar: Jacobians dF/dx [n x n_states x n_states].
            dFdu: Jacobians dF/du [n x n_states x n_inputs].
            eigval_J_xstar: Eigenvalues [n x n_states] (complex).
            eigvec_J_xstar: Eigenvectors [n x n_states x n_states] (complex).
            is_stable: Stability flags [n,].
            cond_id: Condition IDs [n,].
            tol_unique: Tolerance for uniqueness detection.
            dtype: NumPy data type for storage.
        """
        self.xstar = xstar
        self.F_xstar = F_xstar
        self.x_init = x_init
        self.inputs = inputs
        self.qstar = qstar
        self.dq = dq
        self.n_iters = n_iters
        self.J_xstar = J_xstar
        self.dFdu = dFdu
        self.eigval_J_xstar = eigval_J_xstar
        self.eigvec_J_xstar = eigvec_J_xstar
        self.is_stable = is_stable
        self.cond_id = cond_id
        self.tol_unique = float(tol_unique)
        self.dtype = dtype

        # Infer dimensions
        if xstar is not None:
            self.n, self.n_states = xstar.shape
        elif x_init is not None:
            self.n, self.n_states = x_init.shape
        else:
            self.n, self.n_states = 0, 0

        if inputs is not None:
            self.n_inputs = inputs.shape[1]
        else:
            self.n_inputs = 0

    def __len__(self) -> int:
        """Return the number of fixed points."""
        return self.n

    def __getitem__(self, idx):
        """Index into the fixed points.

        Args:
            idx: Integer index, slice, or array of indices.

        Returns:
            A new FixedPoints object containing the indexed subset.
        """
        if isinstance(idx, int):
            idx = [idx]  # Convert to list to preserve dimensionality

        def _index(x):
            return None if x is None else x[idx]

        return FixedPoints(
            xstar=_index(self.xstar),
            F_xstar=_index(self.F_xstar),
            x_init=_index(self.x_init),
            inputs=_index(self.inputs),
            qstar=_index(self.qstar),
            dq=_index(self.dq),
            n_iters=_index(self.n_iters),
            J_xstar=_index(self.J_xstar),
            dFdu=_index(self.dFdu),
            eigval_J_xstar=_index(self.eigval_J_xstar),
            eigvec_J_xstar=_index(self.eigvec_J_xstar),
            is_stable=_index(self.is_stable),
            cond_id=_index(self.cond_id),
            tol_unique=self.tol_unique,
            dtype=self.dtype,
        )

    def get_unique(self):
        """Identify and return unique fixed points.

        Uniqueness is determined by Euclidean distance in the concatenated
        (xstar, inputs) space. Among duplicates, keeps the one with lowest qstar.

        Returns:
            A new FixedPoints object containing only unique fixed points.
        """
        if self.n == 0:
            return self

        # Concatenate xstar and inputs for distance computation
        if self.inputs is None:
            data = self.xstar
        else:
            data = np.concatenate([self.xstar, self.inputs], axis=1)

        idx_keep = []
        idx_checked = np.zeros(self.n, dtype=bool)

        for idx in range(self.n):
            if idx_checked[idx]:
                continue

            # Find all points within tolerance of current point
            dists = np.linalg.norm(data - data[idx], axis=1)
            idx_match = np.where(dists <= self.tol_unique)[0]

            # Among matches, keep the one with smallest qstar
            if len(idx_match) > 1:
                qstars_match = self.qstar[idx_match]
                idx_best = idx_match[np.argmin(qstars_match)]
                idx_keep.append(idx_best)
                idx_checked[idx_match] = True
            else:
                idx_keep.append(idx)
                idx_checked[idx] = True

        return self[idx_keep]

    def decompose_jacobians(self, verbose: bool = False):
        """Compute eigendecomposition of Jacobians and determine stability.

        Computes eigenvalues and eigenvectors for self.J_xstar and determines
        stability based on whether max |eigenvalue| < 1.

        Args:
            verbose: Whether to print status messages.
        """
        if self.J_xstar is None or self.n == 0:
            if verbose:
                print("No Jacobians to decompose.")
            return

        if self.has_decomposed_jacobians:
            if verbose:
                print("Jacobians have already been decomposed.")
            return

        if verbose:
            print(f"Decomposing {self.n} Jacobians...")

        n, n_states, _ = self.J_xstar.shape
        eigvals = np.empty((n, n_states), dtype=np.complex64)
        eigvecs = np.empty((n, n_states, n_states), dtype=np.complex64)
        is_stable = np.zeros(n, dtype=bool)

        for i in range(n):
            # Check for NaNs in Jacobian
            if np.any(np.isnan(self.J_xstar[i])):
                import warnings

                warnings.warn(
                    f"NaNs detected in Jacobian at index {i}. "
                    "This may indicate upstream numerical issues.",
                    stacklevel=2,
                )
                eigvals[i, :] = np.nan
                eigvecs[i, :, :] = np.nan
                is_stable[i] = False
            else:
                vals, vecs = np.linalg.eig(self.J_xstar[i])
                # Sort by magnitude (descending)
                sort_idx = np.argsort(np.abs(vals))[::-1]
                eigvals[i] = vals[sort_idx]
                eigvecs[i] = vecs[:, sort_idx]
                # Stable if largest eigenvalue magnitude < 1
                is_stable[i] = np.abs(vals[sort_idx[0]]) < 1.0

        self.eigval_J_xstar = eigvals
        self.eigvec_J_xstar = eigvecs
        self.is_stable = is_stable

        if verbose:
            n_stable = np.sum(is_stable)
            print(f"Found {n_stable} stable and {n - n_stable} unstable fixed points.")

    @property
    def has_decomposed_jacobians(self) -> bool:
        """Check if Jacobians have been decomposed."""
        return self.eigval_J_xstar is not None

    def print_summary(self):
        """Print a summary of the fixed points."""
        print("\n=== Fixed Points Summary ===")
        print(f"Number of fixed points: {self.n}")
        print(f"State dimension: {self.n_states}")
        print(f"Input dimension: {self.n_inputs}")

        if self.qstar is not None and self.n > 0:
            print(
                f"\nq values: min={np.min(self.qstar):.2e}, "
                f"median={np.median(self.qstar):.2e}, "
                f"max={np.max(self.qstar):.2e}"
            )

        if self.n_iters is not None and self.n > 0:
            print(
                f"Iterations: min={np.min(self.n_iters)}, "
                f"median={np.median(self.n_iters):.0f}, "
                f"max={np.max(self.n_iters)}"
            )

        if self.has_decomposed_jacobians and self.is_stable is not None:
            n_stable = np.sum(self.is_stable)
            print(f"\nStable fixed points: {n_stable} / {self.n}")
