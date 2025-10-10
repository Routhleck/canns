# coding: utf-8
"""FlipFlop memory task with fixed point analysis using BrainState.

This example trains an RNN on a flip-flop memory task and then finds fixed points
in the trained network. The flip-flop task requires the RNN to memorize binary values
across multiple channels, flipping each channel's state when it receives an input pulse.

Based on the PyTorch implementation by Matt Golub.
"""

import brainstate as bst
import jax
import jax.numpy as jnp
import numpy as np
import jax.tree_util as jtu

from canns.analyzer.plotting import plot_fixed_points_2d, plot_fixed_points_3d, PlotConfig
from canns.analyzer.slow_points import FixedPointFinder, save_checkpoint, load_checkpoint


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
        """Forward pass through the RNN. Optimized with jax.lax.scan."""
        batch_size = inputs.shape[0]
        n_time = inputs.shape[1]

        # Initialize hidden state
        if hidden is None:
            h = jnp.tile(self.h0.value, (batch_size, 1))
        else:
            h = hidden

        # Single-step computation mode for the fixed-point finder
        if n_time == 1:
            x_t = inputs[:, 0, :]
            h_next = self.step(x_t, h)
            y = h_next @ self.w_out.value + self.b_out.value
            return y[:, None, :], h_next

        # Full sequence case
        def scan_fn(carry, x_t):
            """Single-step scan function"""
            h_prev = carry
            h_next = self.step(x_t, h_prev)
            y_t = h_next @ self.w_out.value + self.b_out.value
            return h_next, (y_t, h_next)

        # (batch, time, features) -> (time, batch, features)
        inputs_transposed = inputs.transpose(1, 0, 2)

        # Run the scan
        _, (outputs_seq, hiddens_seq) = jax.lax.scan(scan_fn, h, inputs_transposed)

        outputs = outputs_seq.transpose(1, 0, 2)
        hiddens = hiddens_seq.transpose(1, 0, 2)

        return outputs, hiddens


def train_flipflop_rnn(rnn, train_data, valid_data,
                       learning_rate=0.01,
                       batch_size=128,
                       max_epochs=1000,
                       min_loss=1e-4,
                       print_every=10):
    """Train the FlipFlop RNN. Final, robust optimized version with all fixes."""
    print("\n" + "=" * 70)
    print("Training FlipFlop RNN (Final, Robust Optimized Version)")
    print("=" * 70)

    # Prepare data
    train_inputs = jnp.array(train_data['inputs'])
    train_targets = jnp.array(train_data['targets'])
    valid_inputs = jnp.array(valid_data['inputs'])
    valid_targets = jnp.array(valid_data['targets'])
    n_train = train_inputs.shape[0]
    n_batches = n_train // batch_size

    # Convert all parameter keys to strings
    def flatten_key(key):
        return '.'.join(key) if isinstance(key, tuple) else key

    # Flatten the nested keys
    trainable_states = {flatten_key(name): state for name, state in rnn.states().items() if
                        isinstance(state, bst.ParamState)}
    trainable_params = {name: state.value for name, state in trainable_states.items()}

    # Create optimizer
    optimizer = bst.optim.Adam(lr=learning_rate)

    # Register optimizer with flattened-key dictionary
    optimizer.register_trainable_weights(trainable_states)

    # Define JIT-compiled gradient steps
    @jax.jit
    def grad_step(params, batch_inputs, batch_targets):
        """Pure function to compute loss and gradients"""

        def forward_pass(p, inputs):
            batch_size = inputs.shape[0]
            h = jnp.tile(p['h0'], (batch_size, 1))

            def scan_fn(carry, x_t):
                h_prev = carry
                if rnn.rnn_type == "tanh":
                    h_next = jnp.tanh(x_t @ p['w_ih'] + h_prev @ p['w_hh'] + p['b_h'])
                elif rnn.rnn_type == "gru":
                    r = jax.nn.sigmoid(x_t @ p['w_ir'] + h_prev @ p['w_hr'] + p['b_r'])
                    z = jax.nn.sigmoid(x_t @ p['w_iz'] + h_prev @ p['w_hz'] + p['b_z'])
                    n = jnp.tanh(x_t @ p['w_in'] + (r * h_prev) @ p['w_hn'] + p['b_n'])
                    h_next = (1 - z) * n + z * h_prev
                else:
                    h_next = h_prev
                y_t = h_next @ p['w_out'] + p['b_out']
                return h_next, y_t

            inputs_transposed = inputs.transpose(1, 0, 2)
            _, outputs_seq = jax.lax.scan(scan_fn, h, inputs_transposed)
            outputs = outputs_seq.transpose(1, 0, 2)
            return outputs

        def loss_fn(p):
            outputs = forward_pass(p, batch_inputs)
            return jnp.mean((jnp.tanh(outputs) - batch_targets) ** 2)

        loss_val, grads = jax.value_and_grad(loss_fn)(params)
        return loss_val, grads

    def clip_by_global_norm(tree, max_norm=3.0):
        total = jnp.sqrt(sum([jnp.sum(jnp.square(g)) for g in jtu.tree_leaves(tree)]))
        scale = jnp.minimum(1.0, max_norm / (total + 1e-8))
        return jtu.tree_map(lambda g: g * scale, tree)

    # LR scheduling logic
    lr_decay_factor = 0.95
    lr_decay_patience = 20
    best_valid_loss = float('inf')
    patience_counter = 0
    losses = []
    print("\nTraining parameters:")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")

    for epoch in range(max_epochs):
        perm = np.random.permutation(n_train)
        epoch_loss = 0.0
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            batch_inputs = train_inputs[perm[start_idx:end_idx]]
            batch_targets = train_targets[perm[start_idx:end_idx]]

            loss_val, grads = grad_step(trainable_params, batch_inputs, batch_targets)
            grads = clip_by_global_norm(grads, max_norm=3.0)
            optimizer.update(grads)
            # Use flattened keys to update the parameters
            trainable_params = {flatten_key(name): state.value for name, state in rnn.states().items() if
                                isinstance(state, bst.ParamState)}

            epoch_loss += float(loss_val)
        epoch_loss /= n_batches
        losses.append(epoch_loss)
        if epoch % print_every == 0:
            valid_outputs, _ = rnn(valid_inputs)
            valid_loss = float(jnp.mean((valid_outputs - valid_targets) ** 2))
            print(f"Epoch {epoch:4d}: train_loss = {epoch_loss:.6f}, "
                  f"valid_loss = {valid_loss:.6f}, lr = {optimizer.lr.lr:.6f}")

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= lr_decay_patience:
                    optimizer.lr.lr *= lr_decay_factor
                    patience_counter = 0
                    print(f"  -> Reduced learning rate to {optimizer.lr.lr:.6f}")
        if epoch_loss < min_loss:
            print(f"\nReached target loss {min_loss:.2e} at epoch {epoch}")
            break

    valid_outputs, _ = rnn(valid_inputs)
    final_valid_loss = float(jnp.mean((valid_outputs - valid_targets) ** 2))
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Final training loss: {epoch_loss:.6f}")
    print(f"Final validation loss: {final_valid_loss:.6f}")
    print(f"Total epochs: {epoch + 1}")
    return losses


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
    n_time = 64  # Match reference default
    n_trials_train = 512  # Training data size
    n_trials_valid = 128  # Validation data size
    n_trials_test = 128  # Test data size

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
    checkpoint_path = "flipflop_rnn_checkpoint.msgpack"
    if load_checkpoint(rnn, checkpoint_path):
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
            learning_rate=0.08,
            batch_size=128,
            max_epochs=1000,
            min_loss=1e-4,
            print_every=10
        )
        print("\nTraining complete!")

        # Save checkpoint
        save_checkpoint(rnn, checkpoint_path)

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
        lr_init=0.02,  # Match reference
        tol_q=1e-4,  # More relaxed tolerance to stop earlier
        final_q_threshold=1e-6,
        tol_unique=1e-2,  # Larger tolerance to merge nearby fixed points
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
        n_inits=1024,  # Match reference
        noise_scale=0.1,  # Match reference
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
                f"{i + 1:<4} {unique_fps.qstar[i]:<12.2e} {stability_str:<12} {max_eig:<12.4f}"
            )

        # Visualize fixed points in 2D and 3D
        print("\nGenerating 2D visualization...")
        config_2d = PlotConfig(
            title="FlipFlop Fixed Points (2D PCA)",
            xlabel="PC 1",
            ylabel="PC 2",
            figsize=(10, 8),
            save_path="flipflop_fixed_points_2d.png",
            show=False
        )
        plot_fixed_points_2d(unique_fps, hiddens_np, config=config_2d)

        print("\nGenerating 3D visualization...")
        config_3d = PlotConfig(
            title="FlipFlop Fixed Points (3D PCA)",
            figsize=(12, 10),
            save_path="flipflop_fixed_points_3d.png",
            show=False
        )
        plot_fixed_points_3d(
            unique_fps,
            hiddens_np,
            config=config_3d,
            plot_batch_idx=list(range(30)),  # Plot first 30 trajectories
            plot_start_time=10  # Start from timestep 10 to avoid initial transients
        )

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
