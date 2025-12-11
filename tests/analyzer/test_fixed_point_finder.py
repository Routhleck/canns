"""Test script for the fixed point finder with a simple BrainPy RNN.

This example demonstrates how to use the FixedPointFinder with a simple
RNN model to find and analyze fixed points.
"""

import numpy as np
import jax
import jax.numpy as jnp
import brainpy as bp
import brainpy.math as bm
from canns.analyzer.slow_points import FixedPointFinder, FixedPoints


# Define a simple RNN model compatible with BrainPy
class SimpleRNN(bp.DynamicalSystem):
    """Simple vanilla RNN for testing fixed point finder."""

    def __init__(self, n_inputs, n_hidden):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden

        # Initialize parameters
        key = jax.random.PRNGKey(0)
        k1, k2, k3 = jax.random.split(key, 3)

        self.w_ih = bm.Variable(
            jax.random.normal(k1, (n_inputs, n_hidden)) * 0.1
        )
        self.w_hh = bm.Variable(
            jax.random.normal(k2, (n_hidden, n_hidden)) * 0.5
        )
        self.b_h = bm.Variable(jnp.zeros(n_hidden))

    def __call__(self, inputs, hidden):
        """Forward pass.

        Args:
            inputs: [batch_size x time_steps x n_inputs] input sequence.
            hidden: [batch_size x n_hidden] hidden state.

        Returns:
            outputs: [batch_size x time_steps x n_hidden] output sequence.
            h_next: [batch_size x n_hidden] next hidden state.
        """
        # For simplicity, assume time_steps = 1
        # inputs shape: [batch_size x 1 x n_inputs]
        # hidden shape: [batch_size x n_hidden]

        inputs_t = inputs[:, 0, :]  # [batch_size x n_inputs]

        # Compute next hidden state
        h_next = jnp.tanh(
            inputs_t @ self.w_ih.value + hidden @ self.w_hh.value + self.b_h.value
        )

        # For compatibility, return outputs with time dimension
        outputs = h_next[:, None, :]  # [batch_size x 1 x n_hidden]

        return outputs, h_next


def generate_sample_trajectory(rnn, n_batch=32, n_time=100):
    """Generate sample RNN trajectories for testing.

    Args:
        rnn: RNN model.
        n_batch: Number of trajectories.
        n_time: Number of time steps.

    Returns:
        state_traj: [n_batch x n_time x n_hidden] state trajectory.
        inputs: [1 x n_inputs] constant zero input.
    """
    n_hidden = rnn.n_hidden
    n_inputs = rnn.n_inputs

    # Initialize states
    h = jnp.zeros((n_batch, n_hidden))
    state_traj = []

    # Zero input
    u = jnp.zeros((n_batch, 1, n_inputs))

    # Generate trajectory
    for t in range(n_time):
        _, h = rnn(u, h)
        state_traj.append(np.array(h))

    state_traj = np.stack(state_traj, axis=1)  # [n_batch x n_time x n_hidden]

    return state_traj, np.zeros((1, n_inputs), dtype=np.float32)


def main():
    """Main test function."""
    print("=" * 70)
    print("Testing Fixed Point Finder with Simple RNN")
    print("=" * 70)

    # Create a simple RNN
    n_inputs = 2
    n_hidden = 10

    print(f"\nCreating RNN with {n_inputs} inputs and {n_hidden} hidden units...")
    rnn = SimpleRNN(n_inputs, n_hidden)

    # Generate sample trajectories
    print("Generating sample trajectories...")
    state_traj, inputs = generate_sample_trajectory(rnn, n_batch=16, n_time=50)
    print(f"Trajectory shape: {state_traj.shape}")

    # Create fixed point finder
    print("\nInitializing Fixed Point Finder...")
    finder = FixedPointFinder(
        rnn,
        method="joint",
        max_iters=1000,
        tol_q=1e-12,
        do_compute_jacobians=True,
        do_decompose_jacobians=True,
        verbose=True,
        super_verbose=False,
    )

    # Find fixed points
    print("\nSearching for fixed points...")
    unique_fps, all_fps = finder.find_fixed_points(
        state_traj=state_traj,
        inputs=inputs,
        n_inits=128,
        noise_scale=0.1,
    )

    # Print results
    print("\n" + "=" * 70)
    print("Results Summary")
    print("=" * 70)
    unique_fps.print_summary()

    if unique_fps.n > 0:
        print(f"\nFixed points found:")
        for i in range(min(5, unique_fps.n)):  # Show first 5
            stability_str = (
                "stable" if unique_fps.is_stable[i] else "unstable"
            )
            print(f"  FP {i+1}: q = {unique_fps.qstar[i]:.2e}, {stability_str}")
            if unique_fps.eigval_J_xstar is not None:
                max_eig = np.abs(unique_fps.eigval_J_xstar[i, 0])
                print(f"    Max |eigenvalue| = {max_eig:.4f}")

    print("\n" + "=" * 70)
    print("Test completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
