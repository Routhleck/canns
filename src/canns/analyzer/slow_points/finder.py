"""Fixed point finder for BrainPy RNN models."""

import time

import brainpy as bp
import brainpy.math as bm
import jax
import jax.numpy as jnp
import numpy as np

from .fixed_points import FixedPoints


class FixedPointFinder:
    """Find and analyze fixed points in RNN dynamics.

    This class implements an optimization-based approach to finding fixed points
    in recurrent neural networks. It uses gradient descent to minimize the
    objective q = 0.5 * ||x - F(x, u)||^2, where F is the RNN transition function.

    The implementation is compatible with BrainPy RNN models and uses JAX for
    automatic differentiation and optimization.
    """

    def __init__(
        self,
        rnn_model: bp.DynamicalSystem,
        method: str = "joint",
        max_iters: int = 5000,
        tol_q: float = 1e-12,
        tol_dq: float = 1e-20,
        lr_init: float = 1.0,
        lr_factor: float = 0.95,
        lr_patience: int = 5,
        lr_cooldown: int = 0,
        do_compute_jacobians: bool = True,
        do_decompose_jacobians: bool = True,
        tol_unique: float = 1e-3,
        do_exclude_distance_outliers: bool = True,
        outlier_distance_scale: float = 10.0,
        do_rerun_q_outliers: bool = False,
        outlier_q_scale: float = 10.0,
        max_n_unique: float = np.inf,
        final_q_threshold: float = 1e-8,
        dtype: str = "float32",
        verbose: bool = True,
        super_verbose: bool = False,
        n_iters_per_print_update: int = 100,
    ):
        """Initialize the FixedPointFinder.

        Args:
            rnn_model: A BrainPy RNN model with __call__(inputs, hidden) signature.
            method: Optimization method ('joint' or 'sequential').
            max_iters: Maximum optimization iterations.
            tol_q: Tolerance for q value convergence.
            tol_dq: Tolerance for change in q value.
            lr_init: Initial learning rate.
            lr_factor: Learning rate reduction factor.
            lr_patience: Patience for learning rate scheduler.
            lr_cooldown: Cooldown for learning rate scheduler.
            do_compute_jacobians: Whether to compute Jacobians.
            do_decompose_jacobians: Whether to eigendecompose Jacobians.
            tol_unique: Tolerance for identifying unique fixed points.
            do_exclude_distance_outliers: Whether to exclude distance outliers.
            outlier_distance_scale: Scale for distance outlier detection.
            do_rerun_q_outliers: Whether to rerun optimization on q outliers.
            outlier_q_scale: Scale for q outlier detection.
            max_n_unique: Maximum number of unique fixed points to keep.
            dtype: Data type for computations.
            verbose: Print high-level status updates.
            super_verbose: Print per-iteration updates.
            n_iters_per_print_update: Print frequency during optimization.
        """
        self.rnn_model = rnn_model
        self.method = method
        self.max_iters = int(max_iters)
        self.tol_q = float(tol_q)
        self.tol_dq = float(tol_dq)
        self.lr_init = float(lr_init)
        self.lr_factor = float(lr_factor)
        self.lr_patience = int(lr_patience)
        self.lr_cooldown = int(lr_cooldown)
        self.do_compute_jacobians = bool(do_compute_jacobians)
        self.do_decompose_jacobians = bool(do_decompose_jacobians)
        self.tol_unique = float(tol_unique)
        self.do_exclude_distance_outliers = bool(do_exclude_distance_outliers)
        self.outlier_distance_scale = float(outlier_distance_scale)
        self.do_rerun_q_outliers = bool(do_rerun_q_outliers)
        self.outlier_q_scale = float(outlier_q_scale)
        self.max_n_unique = max_n_unique
        self.final_q_threshold = float(final_q_threshold)
        self.verbose = bool(verbose)
        self.super_verbose = bool(super_verbose)
        self.n_iters_per_print_update = int(n_iters_per_print_update)

        # Random number generator
        self.rng = np.random.RandomState(0)

        # Data type
        if dtype == "float32":
            self.np_dtype = np.float32
            self.jax_dtype = jnp.float32
        elif dtype == "float64":
            self.np_dtype = np.float64
            self.jax_dtype = jnp.float64
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

    def find_fixed_points(
        self,
        state_traj: np.ndarray,
        inputs: np.ndarray,
        n_inits: int = 1024,
        noise_scale: float = 0.0,
        valid_bxt: np.ndarray | None = None,
        cond_ids: np.ndarray | None = None,
    ) -> tuple[FixedPoints, FixedPoints]:
        """Find fixed points from sampled RNN states.

        Args:
            state_traj: [n_batch x n_time x n_states] trajectory of RNN states.
            inputs: [1 x n_inputs] or [n_inits x n_inputs] constant inputs.
            n_inits: Number of initial states to sample.
            noise_scale: Std dev of Gaussian noise added to sampled states.
            valid_bxt: [n_batch x n_time] boolean mask for valid samples.
            cond_ids: [n_inits,] condition IDs for each initialization.

        Returns:
            unique_fps: FixedPoints object with unique fixed points.
            all_fps: FixedPoints object with all fixed points before filtering.
        """
        self._print_if_verbose(f"\nSearching for fixed points from {n_inits} initial states.\n")

        # Sample initial states
        initial_states = self._sample_states(state_traj, n_inits, valid_bxt, noise_scale)

        # Prepare inputs
        if inputs.shape[0] == 1:
            inputs_nxd = np.tile(inputs, [n_inits, 1])
        elif inputs.shape[0] == n_inits:
            inputs_nxd = inputs
        else:
            raise ValueError(
                f"Incompatible inputs shape: {inputs.shape}. "
                f"Expected [1, n_inputs] or [{n_inits}, n_inputs]."
            )

        # Run optimization
        if self.method == "joint":
            # Warn if n_inits is large for joint optimization
            LARGE_N_INITS_THRESHOLD = 1000
            if n_inits > LARGE_N_INITS_THRESHOLD:
                import warnings

                warnings.warn(
                    f"Joint optimization with n_inits={n_inits} may be inefficient and use excessive memory. "
                    f"Consider using sequential optimization or reducing n_inits.",
                    stacklevel=2,
                )
            all_fps = self._run_joint_optimization(initial_states, inputs_nxd, cond_ids)
        elif self.method == "sequential":
            all_fps = self._run_sequential_optimizations(initial_states, inputs_nxd, cond_ids)
        else:
            raise ValueError(f"Unsupported method: {self.method}. Must be 'joint' or 'sequential'.")

        # Filter unique fixed points
        unique_fps = all_fps.get_unique()
        self._print_if_verbose(f"\tIdentified {unique_fps.n} unique fixed points.")

        # Exclude distance outliers
        if self.do_exclude_distance_outliers and unique_fps.n > 0:
            unique_fps = self._exclude_distance_outliers(unique_fps, initial_states)

        # Rerun q outliers
        if self.do_rerun_q_outliers and unique_fps.n > 0:
            unique_fps = self._run_additional_iterations_on_outliers(unique_fps)
            unique_fps = unique_fps.get_unique()

        # Limit number of unique fixed points
        if unique_fps.n > self.max_n_unique:
            self._print_if_verbose(
                f"\tSelecting top {int(self.max_n_unique)} unique fixed points by qstar."
            )
            # Sort fixed points by qstar (ascending = better convergence)
            idx_sorted = np.argsort(unique_fps.qstar)
            idx_keep = idx_sorted[: int(self.max_n_unique)]
            unique_fps = unique_fps[idx_keep]

        # Compute Jacobians
        if self.do_compute_jacobians and unique_fps.n > 0:
            self._print_if_verbose(
                f"\tComputing recurrent Jacobian at {unique_fps.n} unique fixed points."
            )
            unique_fps.J_xstar = self._compute_recurrent_jacobians(unique_fps)

            self._print_if_verbose(
                f"\tComputing input Jacobian at {unique_fps.n} unique fixed points."
            )
            unique_fps.dFdu = self._compute_input_jacobians(unique_fps)

            if self.do_decompose_jacobians:
                unique_fps.decompose_jacobians(verbose=self.verbose)

        # Set the conditions for final filtering
        if self.final_q_threshold > 0 and unique_fps.n > 0:
            self._print_if_verbose(
                f"\tApplying final q-value filter (q < {self.final_q_threshold:.1e})..."
            )
            n_before_filter = unique_fps.n

            idx_keep = np.where(unique_fps.qstar < self.final_q_threshold)[0]

            unique_fps = unique_fps[idx_keep]

            n_after_filter = unique_fps.n
            n_discarded = n_before_filter - n_after_filter

            if self.verbose and n_discarded > 0:
                self._print_if_verbose(f"\t\tExcluded {n_discarded} low-quality fixed points.")
            self._print_if_verbose(f"\t\t{n_after_filter} high-quality fixed points remain.")
        self._print_if_verbose("\tFixed point finding complete.\n")

        return unique_fps, all_fps

    def _sample_states(
        self,
        state_traj: np.ndarray,
        n_inits: int,
        valid_bxt: np.ndarray | None,
        noise_scale: float,
    ) -> np.ndarray:
        """Sample initial states from trajectory.

        Args:
            state_traj: [n_batch x n_time x n_states] state trajectory.
            n_inits: Number of samples to draw.
            valid_bxt: [n_batch x n_time] boolean mask.
            noise_scale: Std dev of Gaussian noise.

        Returns:
            [n_inits x n_states] sampled initial states.
        """
        n_batch, n_time, n_states = state_traj.shape

        # Create valid mask
        if valid_bxt is None:
            valid_bxt = np.ones((n_batch, n_time), dtype=bool)
        else:
            assert valid_bxt.shape == (n_batch, n_time), (
                f"valid_bxt shape {valid_bxt.shape} does not match expected ({n_batch}, {n_time})"
            )

        # Sample trial and time indices
        trial_idx, time_idx = np.nonzero(valid_bxt)
        max_sample_index = len(trial_idx)

        # Sample without replacement if possible, otherwise allow duplicates
        if n_inits <= max_sample_index:
            sample_indices = self.rng.choice(max_sample_index, size=n_inits, replace=False)
        else:
            # If we need more samples than available, allow duplicates
            sample_indices = self.rng.randint(max_sample_index, size=n_inits)

        # Draw samples
        states = np.zeros([n_inits, n_states], dtype=self.np_dtype)
        for i, idx in enumerate(sample_indices):
            t_idx = trial_idx[idx]
            time_idx_i = time_idx[idx]
            states[i, :] = state_traj[t_idx, time_idx_i, :]

        # Add noise
        if noise_scale > 0:
            states += noise_scale * self.rng.randn(n_inits, n_states).astype(self.np_dtype)

        assert not np.any(np.isnan(states)), "Detected NaNs in sampled states."

        return states

    def _run_joint_optimization(
        self,
        initial_states: np.ndarray,
        inputs: np.ndarray,
        cond_ids: np.ndarray | None,
    ) -> FixedPoints:
        """Run joint optimization over all initial states.

        Args:
            initial_states: [n x n_states] initial states.
            inputs: [n x n_inputs] constant inputs.
            cond_ids: [n,] condition IDs.

        Returns:
            FixedPoints object with optimization results.
        """
        self._print_if_verbose("\tFinding fixed points via joint optimization.")

        n, n_states = initial_states.shape
        _, n_inputs = inputs.shape

        # Convert to JAX arrays
        x_init = jnp.array(initial_states, dtype=self.jax_dtype)
        u = jnp.array(inputs, dtype=self.jax_dtype)

        # Create optimization variables as BrainPy Variable
        x_state = bm.Variable(x_init)

        # Create optimizer
        optimizer = bp.optim.Adam(lr=self.lr_init)
        optimizer.register_train_vars({"x": x_state})

        # Track learning rate manually (simplified scheduler)
        current_lr = self.lr_init
        lr_patience_counter = 0
        lr_cooldown_counter = 0
        best_q = float("inf")

        # Optimization loop
        iter_count = 0
        q_prev = jnp.full(n, float("nan"))
        t_start = time.time()

        while True:
            iter_count += 1

            # Get current x
            x_current = x_state.value

            # Compute F(x)
            F_x = self._compute_F(x_current, u)

            # Compute q = 0.5 * ||x - F(x)||^2
            dx = x_current - F_x
            q = 0.5 * jnp.sum(dx**2, axis=1)
            q_mean = jnp.mean(q)
            dq = jnp.abs(q - q_prev)

            # Compute gradients
            def loss_fn():
                x_opt = x_state.value
                F_x_opt = self._compute_F(x_opt, u)
                dx_opt = x_opt - F_x_opt
                return jnp.mean(0.5 * jnp.sum(dx_opt**2, axis=1))

            grads_raw = bm.grad(loss_fn, grad_vars=x_state)()

            # Wrap gradients in dictionary for optimizer
            grads = {"x": grads_raw}

            # Update
            optimizer.update(grads)

            # Manual learning rate scheduling (simplified)
            if iter_count > 1:
                if q_mean < best_q:
                    best_q = float(q_mean)
                    lr_patience_counter = 0
                else:
                    if lr_cooldown_counter == 0:
                        lr_patience_counter += 1
                        if lr_patience_counter >= self.lr_patience:
                            current_lr *= self.lr_factor
                            optimizer.lr.value = current_lr
                            lr_patience_counter = 0
                            lr_cooldown_counter = self.lr_cooldown
                    else:
                        lr_cooldown_counter -= 1

            # Print update
            if self.super_verbose and iter_count % self.n_iters_per_print_update == 0:
                self._print_iter_update(
                    iter_count,
                    t_start,
                    np.array(q),
                    np.array(dq),
                    current_lr,
                )

            # Check convergence
            if iter_count > 1 and np.all(
                np.logical_or(
                    np.array(dq) < self.tol_dq * current_lr,
                    np.array(q) < self.tol_q,
                )
            ):
                self._print_if_verbose("\tOptimization complete to desired tolerance.")
                break

            if iter_count >= self.max_iters:
                self._print_if_verbose("\tMaximum iteration count reached. Terminating.")
                break

            q_prev = q

        # Final print
        if self.verbose:
            self._print_iter_update(
                iter_count,
                t_start,
                np.array(q),
                np.array(dq),
                current_lr,
                is_final=True,
            )

        # Extract results
        xstar = np.array(x_state.value, dtype=self.np_dtype)
        F_xstar = np.array(F_x, dtype=self.np_dtype)
        qstar = np.array(q, dtype=self.np_dtype)
        dq_final = np.array(dq, dtype=self.np_dtype)
        n_iters = np.full(n, iter_count, dtype=np.int32)

        return FixedPoints(
            xstar=xstar,
            F_xstar=F_xstar,
            x_init=initial_states.astype(self.np_dtype),
            inputs=inputs.astype(self.np_dtype),
            qstar=qstar,
            dq=dq_final,
            n_iters=n_iters,
            cond_id=cond_ids,
            tol_unique=self.tol_unique,
            dtype=self.np_dtype,
        )

    def _run_sequential_optimizations(
        self,
        initial_states: np.ndarray,
        inputs: np.ndarray,
        cond_ids: np.ndarray | None,
    ) -> FixedPoints:
        """Run sequential optimizations, one initial state at a time.

        Args:
            initial_states: [n x n_states] initial states.
            inputs: [n x n_inputs] constant inputs.
            cond_ids: [n,] condition IDs.

        Returns:
            FixedPoints object with concatenated results.
        """
        self._print_if_verbose("\tFinding fixed points via sequential optimizations...")

        fps_list = []
        n_inits = initial_states.shape[0]

        for i in range(n_inits):
            self._print_if_verbose(f"\n\tInitialization {i + 1} of {n_inits}:")

            cond_id_i = None if cond_ids is None else cond_ids[i : i + 1]

            fps_i = self._run_joint_optimization(
                initial_states[i : i + 1, :],
                inputs[i : i + 1, :],
                cond_id_i,
            )
            fps_list.append(fps_i)

        # Concatenate results
        return self._concatenate_fps(fps_list)

    def _compute_F(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """Compute F(x, u) = next hidden state.

        Args:
            x: [n x n_states] current hidden states.
            u: [n x n_inputs] inputs.

        Returns:
            [n x n_states] next hidden states.
        """
        # Assume the RNN model has signature: output, h_next = model(input, h)
        # We need to expand dims to add time dimension
        u_expanded = jnp.expand_dims(u, axis=1)  # [n x 1 x n_inputs]

        # Call the model
        _, h_next = self.rnn_model(u_expanded, x)

        return h_next

    def _compute_recurrent_jacobians(self, fps: FixedPoints) -> np.ndarray:
        """Compute Jacobian dF/dx at fixed points.

        Args:
            fps: FixedPoints object.

        Returns:
            [n x n_states x n_states] Jacobian matrices.
        """
        xstar = jnp.array(fps.xstar, dtype=self.jax_dtype)
        inputs_jax = jnp.array(fps.inputs, dtype=self.jax_dtype)

        def F_batched(x):
            """Compute F(x) for all x in batch."""
            return self._compute_F(x, inputs_jax)

        # Use JAX vmap + jacrev for efficient batched Jacobian computation
        def jacobian_single(x_i, u_i):
            """Compute Jacobian for a single fixed point."""

            def F_single(x):
                return self._compute_F(x[None, :], u_i[None, :])[0]

            return jax.jacrev(F_single)(x_i)

        jacobian_batched = jax.vmap(jacobian_single)(xstar, inputs_jax)

        return np.array(jacobian_batched, dtype=self.np_dtype)

    def _compute_input_jacobians(self, fps: FixedPoints) -> np.ndarray:
        """Compute Jacobian dF/du at fixed points.

        Args:
            fps: FixedPoints object.

        Returns:
            [n x n_states x n_inputs] Jacobian matrices.
        """
        xstar = jnp.array(fps.xstar, dtype=self.jax_dtype)
        inputs_jax = jnp.array(fps.inputs, dtype=self.jax_dtype)

        def jacobian_single(x_i, u_i):
            """Compute input Jacobian for a single fixed point."""

            def F_single_u(u):
                return self._compute_F(x_i[None, :], u[None, :])[0]

            return jax.jacrev(F_single_u)(u_i)

        jacobian_batched = jax.vmap(jacobian_single)(xstar, inputs_jax)

        return np.array(jacobian_batched, dtype=self.np_dtype)

    def _exclude_distance_outliers(
        self, fps: FixedPoints, initial_states: np.ndarray
    ) -> FixedPoints:
        """Exclude fixed points that are far from initial states.

        Args:
            fps: FixedPoints object.
            initial_states: [n x n_states] initial states.

        Returns:
            FixedPoints object with outliers removed.
        """
        centroid = np.mean(initial_states, axis=0)
        init_dists = np.linalg.norm(initial_states - centroid, axis=1)
        avg_init_dist = np.mean(init_dists) + 1e-12

        fps_dists = np.linalg.norm(fps.xstar - centroid, axis=1)
        scaled_fps_dists = fps_dists / avg_init_dist

        idx_keep = np.where(scaled_fps_dists < self.outlier_distance_scale)[0]

        n_excluded = fps.n - len(idx_keep)
        if self.verbose and n_excluded > 0:
            print(f"\t\tExcluded {n_excluded} distance outliers (of {fps.n}).")

        return fps[idx_keep]

    def _run_additional_iterations_on_outliers(self, fps: FixedPoints) -> FixedPoints:
        """Run additional optimization iterations on q outliers.

        Args:
            fps: FixedPoints object.

        Returns:
            FixedPoints object with improved outlier estimates.
        """
        outlier_min_q = np.median(fps.qstar) * self.outlier_q_scale
        idx_outliers = np.where(fps.qstar > outlier_min_q)[0]

        if len(idx_outliers) == 0:
            return fps

        self._print_if_verbose(
            f"\n\tDetected {len(idx_outliers)} putative q outliers (q > {outlier_min_q:.2e})."
        )
        self._print_if_verbose("\tPerforming sequential optimizations on outliers...")

        outlier_fps = fps[idx_outliers]
        improved_fps = self._run_sequential_optimizations(
            outlier_fps.xstar,
            outlier_fps.inputs,
            outlier_fps.cond_id,
        )

        # Update iteration counts
        improved_fps.n_iters += outlier_fps.n_iters

        # Replace outliers in original fps
        fps.xstar[idx_outliers] = improved_fps.xstar
        fps.F_xstar[idx_outliers] = improved_fps.F_xstar
        fps.qstar[idx_outliers] = improved_fps.qstar
        fps.dq[idx_outliers] = improved_fps.dq
        fps.n_iters[idx_outliers] = improved_fps.n_iters

        return fps

    @staticmethod
    def _concatenate_fps(fps_list) -> FixedPoints:
        """Concatenate a list of FixedPoints objects.

        Args:
            fps_list: List of FixedPoints objects.

        Returns:
            Single concatenated FixedPoints object.
        """
        if len(fps_list) == 0:
            return FixedPoints()

        def cat_attr(attr_name):
            vals = [getattr(fp, attr_name) for fp in fps_list]
            if all(v is None for v in vals):
                return None
            return np.concatenate([v for v in vals if v is not None], axis=0)

        first = fps_list[0]
        return FixedPoints(
            xstar=cat_attr("xstar"),
            F_xstar=cat_attr("F_xstar"),
            x_init=cat_attr("x_init"),
            inputs=cat_attr("inputs"),
            qstar=cat_attr("qstar"),
            dq=cat_attr("dq"),
            n_iters=cat_attr("n_iters"),
            J_xstar=cat_attr("J_xstar"),
            dFdu=cat_attr("dFdu"),
            eigval_J_xstar=cat_attr("eigval_J_xstar"),
            eigvec_J_xstar=cat_attr("eigvec_J_xstar"),
            is_stable=cat_attr("is_stable"),
            cond_id=cat_attr("cond_id"),
            tol_unique=first.tol_unique,
            dtype=first.dtype,
        )

    def _print_if_verbose(self, msg: str):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(msg)

    @staticmethod
    def _print_iter_update(
        iter_count: int,
        t_start: float,
        q: np.ndarray,
        dq: np.ndarray,
        lr: float,
        is_final: bool = False,
    ):
        """Print optimization iteration update."""
        t_elapsed = time.time() - t_start
        avg_iter_time = t_elapsed / iter_count

        if is_final:
            print(f"\t\t{iter_count} iters, ", end="")
        else:
            print(f"\tIter: {iter_count}, ", end="")

        if q.size == 1:
            print(f"q = {q[0]:.2e}, dq = {dq[0]:.2e}, ", end="")
        else:
            print(
                f"q = {np.mean(q):.2e} +/- {np.std(q):.2e}, "
                f"dq = {np.mean(dq):.2e} +/- {np.std(dq):.2e}, ",
                end="",
            )

        print(f"lr = {lr:.2e}, avg iter time = {avg_iter_time:.2e} sec", end="")

        if is_final:
            print()  # Newline
        else:
            print(".")  # Continue line
