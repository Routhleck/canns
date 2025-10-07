# coding: utf-8
"""FlipFlop memory task with fixed point analysis using BrainState.

This example trains an RNN on a flip-flop memory task and then finds fixed points
in the trained network. The flip-flop task requires the RNN to memorize binary values
across multiple channels, flipping each channel's state when it receives an input pulse.

Based on the PyTorch implementation by Matt Golub.
"""

import numpy as np
import jax
import jax.numpy as jnp
import brainstate as bst
import matplotlib.pyplot as plt
import pickle
import os

from canns.analyzer.slow_points import FixedPointFinder


# ============================================================================
# FlipFlop Data Generator
# ============================================================================

class FlipFlopData:
    """Generator for flip-flop memory task data."""

    def __init__(self, n_bits=3, n_time=64, p=0.5, random_seed=0):
        """Initialize FlipFlopData generator.

        Args:
            n_bits: Number of memory channels.
            n_time: Number of timesteps per trial.
            p: Probability of input pulse at each timestep.
            random_seed: Random seed for reproducibility.
        """
        self.rng = np.random.RandomState(random_seed)
        self.n_time = n_time
        self.n_bits = n_bits
        self.p = p

    def generate_data(self, n_trials):
        """Generate flip-flop task data.

        Args:
            n_trials: Number of trials to generate.

        Returns:
            dict with 'inputs' and 'targets' arrays [n_trials x n_time x n_bits].
        """
        n_time = self.n_time
        n_bits = self.n_bits
        p = self.p

        # Generate unsigned input pulses
        unsigned_inputs = self.rng.binomial(1, p, [n_trials, n_time, n_bits])

        # Ensure every trial starts with a pulse
        unsigned_inputs[:, 0, :] = 1

        # Generate random signs {-1, +1}
        random_signs = 2 * self.rng.binomial(1, 0.5, [n_trials, n_time, n_bits]) - 1

        # Apply random signs
        inputs = unsigned_inputs * random_signs

        # Compute targets
        targets = np.zeros([n_trials, n_time, n_bits])
        for trial_idx in range(n_trials):
            for bit_idx in range(n_bits):
                input_seq = inputs[trial_idx, :, bit_idx]
                t_flip = np.where(input_seq != 0)[0]
                for flip_idx in range(len(t_flip)):
                    t_flip_i = t_flip[flip_idx]
                    targets[trial_idx, t_flip_i:, bit_idx] = inputs[
                        trial_idx, t_flip_i, bit_idx
                    ]

        return {
            "inputs": inputs.astype(np.float32),
            "targets": targets.astype(np.float32),
        }


# ============================================================================
# FlipFlop RNN Model
# ============================================================================

class FlipFlopRNN(bst.nn.Module):
    """RNN model for the flip-flop memory task."""

    def __init__(self, n_inputs, n_hidden, n_outputs, rnn_type="gru", seed=0):
        """Initialize FlipFlop RNN.

        Args:
            n_inputs: Number of input channels.
            n_hidden: Number of hidden units.
            n_outputs: Number of output channels.
            rnn_type: Type of RNN cell ('tanh', 'gru').
            seed: Random seed for weight initialization.
        """
        super().__init__()
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.rnn_type = rnn_type.lower()

        # Initialize RNN cell parameters
        key = jax.random.PRNGKey(seed)
        k1, k2, k3, k4 = jax.random.split(key, 4)

        if rnn_type == "tanh":
            # Simple tanh RNN
            self.w_ih = bst.ParamState(
                jax.random.normal(k1, (n_inputs, n_hidden)) * 0.1
            )
            self.w_hh = bst.ParamState(
                jax.random.normal(k2, (n_hidden, n_hidden)) * 0.5
            )
            self.b_h = bst.ParamState(jnp.zeros(n_hidden))
        elif rnn_type == "gru":
            # GRU cell
            self.w_ir = bst.ParamState(
                jax.random.normal(k1, (n_inputs, n_hidden)) * 0.1
            )
            self.w_hr = bst.ParamState(
                jax.random.normal(k2, (n_hidden, n_hidden)) * 0.5
            )
            self.w_iz = bst.ParamState(
                jax.random.normal(k3, (n_inputs, n_hidden)) * 0.1
            )
            self.w_hz = bst.ParamState(
                jax.random.normal(k4, (n_hidden, n_hidden)) * 0.5
            )
            k5, k6, k7, k8 = jax.random.split(k4, 4)
            self.w_in = bst.ParamState(
                jax.random.normal(k5, (n_inputs, n_hidden)) * 0.1
            )
            self.w_hn = bst.ParamState(
                jax.random.normal(k6, (n_hidden, n_hidden)) * 0.5
            )
            self.b_r = bst.ParamState(jnp.zeros(n_hidden))
            self.b_z = bst.ParamState(jnp.zeros(n_hidden))
            self.b_n = bst.ParamState(jnp.zeros(n_hidden))
        else:
            raise ValueError(f"Unsupported rnn_type: {rnn_type}")

        # Readout layer
        self.w_out = bst.ParamState(
            jax.random.normal(k3, (n_hidden, n_outputs)) * 0.1
        )
        self.b_out = bst.ParamState(jnp.zeros(n_outputs))

        # Initial hidden state
        self.h0 = bst.ParamState(jnp.zeros(n_hidden))

    def step(self, x_t, h):
        """Single RNN step.

        Args:
            x_t: [batch_size x n_inputs] input at time t.
            h: [batch_size x n_hidden] hidden state.

        Returns:
            h_next: [batch_size x n_hidden] next hidden state.
        """
        if self.rnn_type == "tanh":
            # Simple tanh RNN step
            h_next = jnp.tanh(
                x_t @ self.w_ih.value + h @ self.w_hh.value + self.b_h.value
            )
        elif self.rnn_type == "gru":
            # GRU step
            r = jax.nn.sigmoid(
                x_t @ self.w_ir.value + h @ self.w_hr.value + self.b_r.value
            )
            z = jax.nn.sigmoid(
                x_t @ self.w_iz.value + h @ self.w_hz.value + self.b_z.value
            )
            n = jnp.tanh(
                x_t @ self.w_in.value + (r * h) @ self.w_hn.value + self.b_n.value
            )
            h_next = (1 - z) * n + z * h
        else:
            raise ValueError(f"Unknown rnn_type: {self.rnn_type}")

        return h_next

    def __call__(self, inputs, hidden=None):
        """Forward pass through the RNN.

        This can handle two cases:
        1. Full sequence: inputs [batch x time x input_dim] -> outputs [batch x time x output], hiddens [batch x time x hidden]
        2. Single step (for fixed point finder): inputs [batch x 1 x input_dim], hidden [batch x hidden] -> output, h_next

        Args:
            inputs: [batch_size x n_time x n_inputs] input sequence.
            hidden: [batch_size x n_hidden] initial hidden state (optional).

        Returns:
            outputs: [batch_size x n_time x n_outputs] output sequence.
            hiddens: [batch_size x (n_time or 1) x n_hidden] or [batch x hidden] hidden state.
        """
        batch_size = inputs.shape[0]
        n_time = inputs.shape[1]

        # Initialize hidden state
        if hidden is None:
            h = jnp.tile(self.h0.value, (batch_size, 1))
        else:
            h = hidden

        # Single step case (for fixed point finder)
        if n_time == 1:
            x_t = inputs[:, 0, :]
            h_next = self.step(x_t, h)
            # Return output and next hidden (without time dimension)
            y = h_next @ self.w_out.value + self.b_out.value
            return y[:, None, :], h_next

        # Full sequence case
        outputs_list = []
        hiddens_list = []

        for t in range(n_time):
            x_t = inputs[:, t, :]
            h = self.step(x_t, h)

            # Compute output
            y_t = h @ self.w_out.value + self.b_out.value

            outputs_list.append(y_t[:, None, :])
            hiddens_list.append(h[:, None, :])

        outputs = jnp.concatenate(outputs_list, axis=1)
        hiddens = jnp.concatenate(hiddens_list, axis=1)

        return outputs, hiddens


# ============================================================================
# Fixed Point Analysis Utilities
# ============================================================================

def plot_fixed_points_2d(unique_fps, state_traj, title="Fixed Points Analysis (2D)"):
    """Visualize fixed points using 2D PCA."""
    if unique_fps.n == 0:
        print("No fixed points found to plot.")
        return

    from sklearn.decomposition import PCA

    # Flatten trajectories for PCA
    traj_flat = state_traj.reshape(-1, state_traj.shape[2])

    # Fit PCA
    pca = PCA(n_components=2)
    traj_pca = pca.fit_transform(traj_flat)
    fps_pca = pca.transform(unique_fps.xstar)

    # Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(
        traj_pca[:, 0],
        traj_pca[:, 1],
        c="lightblue",
        alpha=0.1,
        label="State Trajectories",
        s=1,
    )

    if unique_fps.is_stable is not None:
        stable_idx = np.where(unique_fps.is_stable)[0]
        unstable_idx = np.where(~unique_fps.is_stable)[0]

        if len(stable_idx) > 0:
            plt.scatter(
                fps_pca[stable_idx, 0],
                fps_pca[stable_idx, 1],
                c="green",
                marker="o",
                s=200,
                label="Stable Fixed Points",
                edgecolors="black",
                linewidths=2,
            )

        if len(unstable_idx) > 0:
            plt.scatter(
                fps_pca[unstable_idx, 0],
                fps_pca[unstable_idx, 1],
                c="red",
                marker="x",
                s=200,
                label="Unstable Fixed Points",
                linewidths=3,
            )
    else:
        plt.scatter(
            fps_pca[:, 0],
            fps_pca[:, 1],
            c="purple",
            marker="*",
            s=200,
            label="Fixed Points",
        )

    plt.xlabel("PC 1", fontsize=12, fontweight="bold")
    plt.ylabel("PC 2", fontsize=12, fontweight="bold")
    plt.title(title, fontsize=14, fontweight="bold")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("flipflop_fixed_points_2d.png", dpi=150, bbox_inches="tight")
    print(f"  Saved 2D plot to: flipflop_fixed_points_2d.png")


def plot_fixed_points_3d(unique_fps, state_traj, title="Fixed Points Analysis (3D)",
                         plot_batch_idx=None, plot_start_time=0):
    """Visualize fixed points using 3D PCA.

    Args:
        unique_fps: FixedPoints object with the fixed points.
        state_traj: [n_batch x n_time x n_states] array of state trajectories.
        title: Plot title.
        plot_batch_idx: List of batch indices to plot trajectories for (default: first 30).
        plot_start_time: Time index to start plotting trajectories from.
    """
    if unique_fps.n == 0:
        print("No fixed points found to plot.")
        return

    from sklearn.decomposition import PCA
    from mpl_toolkits.mplot3d import Axes3D

    # Flatten trajectories for PCA
    traj_flat = state_traj.reshape(-1, state_traj.shape[2])

    # Fit 3D PCA
    pca = PCA(n_components=3)
    traj_pca = pca.fit_transform(traj_flat)
    fps_pca = pca.transform(unique_fps.xstar)

    print(f"  PCA explained variance: {pca.explained_variance_ratio_[:3]}")
    print(f"  Total variance explained: {pca.explained_variance_ratio_[:3].sum():.2%}")

    # Create 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot individual trajectories as lines
    if plot_batch_idx is None:
        plot_batch_idx = list(range(min(30, state_traj.shape[0])))

    n_batch, n_time, n_states = state_traj.shape

    for batch_idx in plot_batch_idx:
        # Transform this trajectory
        traj_single = state_traj[batch_idx, plot_start_time:, :]
        traj_single_pca = pca.transform(traj_single)

        # Plot as a line
        ax.plot(
            traj_single_pca[:, 0],
            traj_single_pca[:, 1],
            traj_single_pca[:, 2],
            c='blue',
            alpha=0.3,
            linewidth=0.5,
        )

    # Plot fixed points
    if unique_fps.is_stable is not None:
        stable_idx = np.where(unique_fps.is_stable)[0]
        unstable_idx = np.where(~unique_fps.is_stable)[0]

        if len(stable_idx) > 0:
            ax.scatter(
                fps_pca[stable_idx, 0],
                fps_pca[stable_idx, 1],
                fps_pca[stable_idx, 2],
                c='black',
                marker='o',
                s=100,
                label=f"Stable FPs ({len(stable_idx)})",
                edgecolors='white',
                linewidths=2,
            )

        if len(unstable_idx) > 0:
            ax.scatter(
                fps_pca[unstable_idx, 0],
                fps_pca[unstable_idx, 1],
                fps_pca[unstable_idx, 2],
                c='red',
                marker='x',
                s=80,
                label=f"Unstable FPs ({len(unstable_idx)})",
                linewidths=2,
            )
    else:
        ax.scatter(
            fps_pca[:, 0],
            fps_pca[:, 1],
            fps_pca[:, 2],
            c='purple',
            marker='*',
            s=100,
            label=f"Fixed Points ({unique_fps.n})",
        )

    ax.set_xlabel('PC 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('PC 2', fontsize=12, fontweight='bold')
    ax.set_zlabel('PC 3', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Set equal aspect ratio
    max_range = np.array([
        fps_pca[:, 0].max() - fps_pca[:, 0].min(),
        fps_pca[:, 1].max() - fps_pca[:, 1].min(),
        fps_pca[:, 2].max() - fps_pca[:, 2].min()
    ]).max() / 2.0

    mid_x = (fps_pca[:, 0].max() + fps_pca[:, 0].min()) * 0.5
    mid_y = (fps_pca[:, 1].max() + fps_pca[:, 1].min()) * 0.5
    mid_z = (fps_pca[:, 2].max() + fps_pca[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()
    plt.savefig("flipflop_fixed_points_3d.png", dpi=150, bbox_inches="tight")
    print(f"  Saved 3D plot to: flipflop_fixed_points_3d.png")

    return fig


# ============================================================================
# Training Functions
# ============================================================================

def train_flipflop_rnn(rnn, train_data, valid_data,
                       learning_rate=0.01,
                       batch_size=128,
                       max_epochs=1000,
                       min_loss=1e-4,
                       print_every=10):
    """Train the FlipFlop RNN.

    Args:
        rnn: FlipFlop RNN model.
        train_data: Training data dict with 'inputs' and 'targets'.
        valid_data: Validation data dict.
        learning_rate: Initial learning rate.
        batch_size: Batch size for training.
        max_epochs: Maximum number of epochs.
        min_loss: Target loss to stop training.
        print_every: Print frequency.

    Returns:
        losses: List of training losses.
    """
    print("\n" + "=" * 70)
    print("Training FlipFlop RNN")
    print("=" * 70)

    # Prepare data
    train_inputs = jnp.array(train_data['inputs'])
    train_targets = jnp.array(train_data['targets'])
    valid_inputs = jnp.array(valid_data['inputs'])
    valid_targets = jnp.array(valid_data['targets'])

    n_train = train_inputs.shape[0]
    n_batches = n_train // batch_size

    # Create optimizer
    optimizer = bst.optim.Adam(lr=learning_rate)

    # Get all trainable parameters
    trainable_params = {}
    for name, state in rnn.states().items():
        if isinstance(state, bst.ParamState):
            trainable_params[name] = state

    optimizer.register_trainable_weights(trainable_params)

    # Learning rate scheduler (simple exponential decay)
    lr_decay_factor = 0.95
    lr_decay_patience = 20
    best_valid_loss = float('inf')
    patience_counter = 0

    losses = []

    print(f"\nTraining parameters:")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Max epochs: {max_epochs}")
    print(f"  Target loss: {min_loss:.2e}")
    print(f"  Number of trainable params: {len(trainable_params)}")

    # Training loop
    for epoch in range(max_epochs):
        # Shuffle training data
        perm = np.random.permutation(n_train)
        train_inputs_shuffled = train_inputs[perm]
        train_targets_shuffled = train_targets[perm]

        epoch_loss = 0.0

        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size

            batch_inputs = train_inputs_shuffled[start_idx:end_idx]
            batch_targets = train_targets_shuffled[start_idx:end_idx]

            # Define loss function
            def loss_fn():
                outputs, _ = rnn(batch_inputs)
                # MSE loss
                return jnp.mean((outputs - batch_targets) ** 2)

            # Compute loss and gradients
            loss_val = loss_fn()
            grads = bst.augment.grad(loss_fn, grad_states=trainable_params)()

            # Update parameters (grads is already a dict matching trainable_params)
            optimizer.update(grads)

            epoch_loss += float(loss_val)

        # Average loss over batches
        epoch_loss /= n_batches
        losses.append(epoch_loss)

        # Validate
        if epoch % print_every == 0:
            valid_outputs, _ = rnn(valid_inputs)
            valid_loss = float(jnp.mean((valid_outputs - valid_targets) ** 2))

            print(f"Epoch {epoch:4d}: train_loss = {epoch_loss:.6f}, "
                  f"valid_loss = {valid_loss:.6f}, lr = {optimizer.lr():.6f}")

            # Learning rate scheduling
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= lr_decay_patience:
                    optimizer.lr._lr *= lr_decay_factor
                    patience_counter = 0
                    print(f"  -> Reduced learning rate to {optimizer.lr():.6f}")

        # Check convergence
        if epoch_loss < min_loss:
            print(f"\nReached target loss {min_loss:.2e} at epoch {epoch}")
            break

    # Final validation
    valid_outputs, _ = rnn(valid_inputs)
    final_valid_loss = float(jnp.mean((valid_outputs - valid_targets) ** 2))

    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Final training loss: {epoch_loss:.6f}")
    print(f"Final validation loss: {final_valid_loss:.6f}")
    print(f"Total epochs: {epoch + 1}")

    return losses


def save_rnn_checkpoint(rnn, filepath="flipflop_rnn_checkpoint.pkl"):
    """Save RNN model parameters to a checkpoint file.

    Args:
        rnn: FlipFlopRNN model to save.
        filepath: Path to save the checkpoint file.
    """
    # Extract all parameter states
    params = {}
    for name_tuple, state in rnn.states().items():
        if isinstance(state, bst.ParamState):
            # Convert tuple key to string for easier handling
            if isinstance(name_tuple, tuple):
                name = name_tuple[0]  # Extract string from tuple
            else:
                name = name_tuple
            params[name] = np.array(state.value)

    with open(filepath, 'wb') as f:
        pickle.dump(params, f)
    print(f"Saved checkpoint to: {filepath}")


def load_rnn_checkpoint(rnn, filepath="flipflop_rnn_checkpoint.pkl"):
    """Load RNN model parameters from a checkpoint file.

    Args:
        rnn: FlipFlopRNN model to load parameters into.
        filepath: Path to the checkpoint file.

    Returns:
        True if checkpoint was loaded successfully, False otherwise.
    """
    if not os.path.exists(filepath):
        return False

    with open(filepath, 'rb') as f:
        params = pickle.load(f)

    # Load parameters into model - match with states dictionary
    states_dict = rnn.states()
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


# ============================================================================
# Main Script
# ============================================================================

def main(seed=42):
    """Main function to train RNN and find fixed points.

    This example trains a RNN to solve the FlipFlop task, then analyzes its
    fixed points. A properly trained 3-bit FlipFlop RNN should have:
    - 8 stable fixed points (2^3 memory states at cube corners)
    - Unstable fixed points on edges/faces/center of the cube
    - Clean separation between stable and unstable points

    Args:
        seed: Random seed for reproducibility (default: 0)
    """
    # Set random seeds for reproducibility
    np.random.seed(seed)
    import random
    random.seed(seed)
    # JAX uses a different random system (PRNG key), but we set numpy seed
    # which affects data generation

    print("=" * 70)
    print("FlipFlop Memory Task - Training and Fixed Point Analysis")
    print("=" * 70)
    print(f"Random seed: {seed}")

    # Task parameters (matching reference implementation)
    n_bits = 3
    n_hidden = 16  # Match reference: 16 hidden units
    n_time = 64    # Match reference default
    n_trials_train = 512  # Training data size
    n_trials_valid = 128  # Validation data size
    n_trials_test = 128   # Test data size

    print(f"\nTask Parameters:")
    print(f"  Number of bits: {n_bits}")
    print(f"  Hidden units: {n_hidden}")
    print(f"  Sequence length: {n_time}")
    print(f"  RNN type: tanh (GRU has issues with FP finding in PyTorch/JAX)")
    print(f"  Training trials: {n_trials_train}")
    print(f"  Validation trials: {n_trials_valid}")

    # Generate data
    print("\nGenerating FlipFlop data...")
    data_gen = FlipFlopData(n_bits=n_bits, n_time=n_time, p=0.5, random_seed=seed)
    train_data = data_gen.generate_data(n_trials_train)
    valid_data = data_gen.generate_data(n_trials_valid)
    test_data = data_gen.generate_data(n_trials_test)

    print(f"  Train inputs shape: {train_data['inputs'].shape}")
    print(f"  Train targets shape: {train_data['targets'].shape}")
    print(f"  Valid inputs shape: {valid_data['inputs'].shape}")
    print(f"  Test inputs shape: {test_data['inputs'].shape}")

    # Create RNN model (use tanh, not GRU)
    print("\nCreating RNN model...")
    rnn = FlipFlopRNN(n_inputs=n_bits, n_hidden=n_hidden, n_outputs=n_bits, rnn_type="tanh", seed=seed)

    # Check for checkpoint
    checkpoint_path = "flipflop_rnn_checkpoint.pkl"
    if load_rnn_checkpoint(rnn, checkpoint_path):
        print("Using pre-trained model from checkpoint.")
    else:
        # Train the RNN
        print("\n" + "=" * 70)
        print("Training RNN on FlipFlop Task")
        print("=" * 70)
        print("No checkpoint found. Training from scratch...")
        losses = train_flipflop_rnn(
            rnn,
            train_data,
            valid_data,
            learning_rate=0.05,
            batch_size=128,
            max_epochs=200,
            min_loss=1e-4,
            print_every=10
        )
        print("\nTraining complete!")

        # Save checkpoint
        save_rnn_checkpoint(rnn, checkpoint_path)

    # Generate trajectories for fixed point analysis using trained RNN
    print("\n" + "=" * 70)
    print("Fixed Point Analysis")
    print("=" * 70)
    print("\nGenerating RNN trajectories with trained model...")
    inputs_jax = jnp.array(test_data["inputs"])
    outputs, hiddens = rnn(inputs_jax)

    hiddens_np = np.array(hiddens)
    print(f"  Hidden state trajectories shape: {hiddens_np.shape}")

    # Find fixed points (matching reference hyperparameters)
    print("\nInitializing Fixed Point Finder...")
    finder = FixedPointFinder(
        rnn,
        method="joint",
        max_iters=10000,  # Match reference
        lr_init=1.0,      # Match reference
        tol_q=1e-6,       # More relaxed tolerance to stop earlier
        tol_unique=0.5,   # Larger tolerance to merge nearby fixed points
        do_compute_jacobians=True,
        do_decompose_jacobians=True,
        outlier_distance_scale=10.0,  # Match reference
        verbose=True,
        super_verbose=True,  # Match reference to see iteration updates
    )

    print("\nSearching for fixed points...")
    # Study system with zero input (like reference)
    constant_input = np.zeros((1, n_bits), dtype=np.float32)

    unique_fps, all_fps = finder.find_fixed_points(
        state_traj=hiddens_np,
        inputs=constant_input,
        n_inits=1024,      # Match reference
        noise_scale=0.5,   # Match reference
    )

    # Print results
    print("\n" + "=" * 70)
    print("Fixed Point Analysis Results")
    print("=" * 70)
    unique_fps.print_summary()

    if unique_fps.n > 0:
        print(f"\nDetailed Fixed Point Information:")
        print(f"{'#':<4} {'q-value':<12} {'Stability':<12} {'Max |eig|':<12}")
        print("-" * 45)
        for i in range(min(10, unique_fps.n)):
            stability_str = "Stable" if unique_fps.is_stable[i] else "Unstable"
            max_eig = np.abs(unique_fps.eigval_J_xstar[i, 0])
            print(
                f"{i+1:<4} {unique_fps.qstar[i]:<12.2e} {stability_str:<12} {max_eig:<12.4f}"
            )

        # Visualize fixed points in 2D and 3D
        print("\nGenerating 2D visualization...")
        plot_fixed_points_2d(unique_fps, hiddens_np, title="FlipFlop Fixed Points (2D PCA)")

        print("\nGenerating 3D visualization...")
        plot_fixed_points_3d(
            unique_fps,
            hiddens_np,
            title="FlipFlop Fixed Points (3D PCA)",
            plot_batch_idx=list(range(30)),  # Plot first 30 trajectories
            plot_start_time=10  # Start from timestep 10 to avoid initial transients
        )

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
