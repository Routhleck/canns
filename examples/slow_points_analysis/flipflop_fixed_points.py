# coding: utf-8
"""FlipFlop memory task with fixed point analysis using BrainState.

This example trains an RNN on a flip-flop memory task and then finds fixed points
in the trained network. The flip-flop task requires the RNN to memorize binary values
across multiple channels, flipping each channel's state when it receives an input pulse.

Based on the PyTorch implementation by Matt Golub.
"""

import random

import brainpy as bp
import brainpy.math as bm
import jax
import jax.numpy as jnp
import numpy as np

from canns.analyzer.visualization import PlotConfig
from canns.analyzer.slow_points import FixedPointFinder, load_checkpoint, plot_fixed_points_2d, \
    plot_fixed_points_3d


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


class FlipFlopRNN(bp.DynamicalSystem):
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
            self.w_ih = bm.Variable(
                jax.random.normal(k1, (n_inputs, n_hidden)) * 0.1
            )
            self.w_hh = bm.Variable(
                jax.random.normal(k2, (n_hidden, n_hidden)) * 0.5
            )
            self.b_h = bm.Variable(jnp.zeros(n_hidden))
        elif rnn_type == "gru":
            # GRU cell
            self.w_ir = bm.Variable(
                jax.random.normal(k1, (n_inputs, n_hidden)) * 0.1
            )
            self.w_hr = bm.Variable(
                jax.random.normal(k2, (n_hidden, n_hidden)) * 0.5
            )
            self.w_iz = bm.Variable(
                jax.random.normal(k3, (n_inputs, n_hidden)) * 0.1
            )
            self.w_hz = bm.Variable(
                jax.random.normal(k4, (n_hidden, n_hidden)) * 0.5
            )
            k5, k6, k7, k8 = jax.random.split(k4, 4)
            self.w_in = bm.Variable(
                jax.random.normal(k5, (n_inputs, n_hidden)) * 0.1
            )
            self.w_hn = bm.Variable(
                jax.random.normal(k6, (n_hidden, n_hidden)) * 0.5
            )
            self.b_r = bm.Variable(jnp.zeros(n_hidden))
            self.b_z = bm.Variable(jnp.zeros(n_hidden))
            self.b_n = bm.Variable(jnp.zeros(n_hidden))
        else:
            raise ValueError(f"Unsupported rnn_type: {rnn_type}")

        # Readout layer
        self.w_out = bm.Variable(
            jax.random.normal(k3, (n_hidden, n_outputs)) * 0.1
        )
        self.b_out = bm.Variable(jnp.zeros(n_outputs))

        # Initial hidden state
        self.h0 = bm.Variable(jnp.zeros(n_hidden))

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
        h = jnp.tile(self.h0.value, (batch_size, 1)) if hidden is None else hidden

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
                       learning_rate=0.08,
                       batch_size=128,
                       max_epochs=1000,
                       min_loss=1e-4,
                       print_every=10):
    print("\n" + "=" * 70)
    print("Training FlipFlop RNN (Using brainpy optimizer)")
    print("=" * 70)

    # Prepare data
    train_inputs = jnp.array(train_data['inputs'])
    train_targets = jnp.array(train_data['targets'])
    valid_inputs = jnp.array(valid_data['inputs'])
    valid_targets = jnp.array(valid_data['targets'])
    n_train = train_inputs.shape[0]
    n_batches = n_train // batch_size

    # Get trainable variables from the model
    # Note: vars() returns keys like 'FlipFlopRNN0.w_ih', we need just 'w_ih' for computation
    train_vars = {name: var for name, var in rnn.vars().items() if isinstance(var, bm.Variable)}
    # Create mapping between short names and full names
    name_mapping = {name.split('.')[-1]: name for name in train_vars.keys()}
    # Extract just the parameter name (after the last dot) for gradient computation
    params = {name.split('.')[-1]: var.value for name, var in train_vars.items()}

    # Initialize optimizer with train_vars parameter (modern brainpy API)
    optimizer = bp.optimizers.Adam(lr=learning_rate, train_vars=train_vars)

    # Define JIT-compiled gradient step
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
            return jnp.mean((outputs - batch_targets) ** 2)

        loss_val, grads = jax.value_and_grad(loss_fn)(params)
        return loss_val, grads

    losses = []
    print("\nTraining parameters:")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate:{learning_rate:.6f} (Fixed)")

    for epoch in range(max_epochs):
        perm = np.random.permutation(n_train)
        epoch_loss = 0.0
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            batch_inputs = train_inputs[perm[start_idx:end_idx]]
            batch_targets = train_targets[perm[start_idx:end_idx]]
            loss_val, grads_short = grad_step(params, batch_inputs, batch_targets)
            # Map gradients back to full names for optimizer
            grads = {name_mapping[short_name]: grad for short_name, grad in grads_short.items()}
            optimizer.update(grads)
            # Update params with current variable values (extract parameter names)
            params = {name.split('.')[-1]: var.value for name, var in train_vars.items()}
            epoch_loss += float(loss_val)
        epoch_loss /= n_batches
        losses.append(epoch_loss)

        if epoch % print_every == 0:
            valid_outputs, _ = rnn(valid_inputs)
            valid_loss = float(jnp.mean((valid_outputs - valid_targets) ** 2))
            print(f"Epoch {epoch:4d}: train_loss = {epoch_loss:.6f}, "
                  f"valid_loss = {valid_loss:.6f}")
        if epoch_loss < min_loss:
            print(f"\nReached target loss {min_loss:.2e} at epoch {epoch}")
            break

    # Training complete
    valid_outputs, _ = rnn(valid_inputs)
    final_valid_loss = float(jnp.mean((valid_outputs - valid_targets) ** 2))
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Final training loss: {epoch_loss:.6f}")
    print(f"Final validation loss: {final_valid_loss:.6f}")
    print(f"Total epochs: {epoch + 1}")
    return losses


# Configuration Dictionary
TASK_CONFIGS = {
    "2_bit": {
        "n_bits": 2,
        "n_hidden": 3,
        "n_trials_train": 512,
        "n_inits": 1024,
    },
    "3_bit": {
        "n_bits": 3,
        "n_hidden": 4,
        "n_trials_train": 512,
        "n_inits": 1024,
    },
    "4_bit": {
        "n_bits": 4,
        "n_hidden": 6,
        "n_trials_train": 512,
        "n_inits": 1024,
    },
}


# seed,7582,8356,9071,
def main(config_name="4_bit", seed=np.random.randint(1, 10000)):
    """Main function to train RNN and find fixed points."""
    # Get configuration from dictionary
    if config_name not in TASK_CONFIGS:
        raise ValueError(f"Unknown config_name: {config_name}. Available: {list(TASK_CONFIGS.keys())}")
    config = TASK_CONFIGS[config_name]

    # Set random seeds
    np.random.seed(seed)
    random.seed(seed)

    print(f"\n--- Running FlipFlop Task ({config_name}) ---")
    print(f"Seed: {seed}")

    n_bits = config["n_bits"]
    n_hidden = config["n_hidden"]
    n_trials_train = config["n_trials_train"]
    n_inits = config["n_inits"]

    n_time = 64
    n_trials_valid = 128
    n_trials_test = 128
    rnn_type = "tanh"
    learning_rate = 0.08
    batch_size = 128
    max_epochs = 500
    min_loss = 1e-4

    # Generate data
    data_gen = FlipFlopData(n_bits=n_bits, n_time=n_time, p=0.5, random_seed=seed)
    train_data = data_gen.generate_data(n_trials_train)
    valid_data = data_gen.generate_data(n_trials_valid)
    test_data = data_gen.generate_data(n_trials_test)

    # Create RNN model
    rnn = FlipFlopRNN(n_inputs=n_bits, n_hidden=n_hidden, n_outputs=n_bits, rnn_type=rnn_type, seed=seed)

    # Check for checkpoint
    checkpoint_path = f"flipflop_rnn_{config_name}_checkpoint.msgpack"
    if load_checkpoint(rnn, checkpoint_path):
        print(f"Loaded model from checkpoint: {checkpoint_path}")
    else:
        # Train the RNN
        print(f"No checkpoint found ({checkpoint_path}). Training...")
        losses = train_flipflop_rnn(
            rnn,
            train_data,
            valid_data,
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_epochs=max_epochs,
            min_loss=min_loss,
            print_every=10
        )

    # Fixed Point Analysis
    print("\n--- Fixed Point Analysis ---")
    inputs_jax = jnp.array(test_data["inputs"])
    outputs, hiddens = rnn(inputs_jax)
    hiddens_np = np.array(hiddens)

    # Find fixed points
    finder = FixedPointFinder(
        rnn,
        method="joint",
        max_iters=5000,
        lr_init=0.02,
        tol_q=1e-4,
        final_q_threshold=1e-6,
        tol_unique=1e-2,
        do_compute_jacobians=True,
        do_decompose_jacobians=True,
        outlier_distance_scale=10.0,
        verbose=True,
        super_verbose=True,
    )

    constant_input = np.zeros((1, n_bits), dtype=np.float32)

    unique_fps, all_fps = finder.find_fixed_points(
        state_traj=hiddens_np,
        inputs=constant_input,
        n_inits=n_inits,
        noise_scale=0.4,
    )

    # Print results
    print("\n--- Fixed Point Analysis Results ---")
    unique_fps.print_summary()

    if unique_fps.n > 0:
        print(f"\nDetailed Fixed Point Information (Top 10):")
        print(f"{'#':<4} {'q-value':<12} {'Stability':<12} {'Max |eig|':<12}")
        print("-" * 45)
        for i in range(min(10, unique_fps.n)):
            stability_str = "Stable" if unique_fps.is_stable[i] else "Unstable"
            max_eig = np.abs(unique_fps.eigval_J_xstar[i, 0])
            print(
                f"{i + 1:<4} {unique_fps.qstar[i]:<12.2e} {stability_str:<12} {max_eig:<12.4f}"
            )

        # Visualize fixed points
        save_path_2d = f"flipflop_{config_name}_fixed_points_2d.png"
        config_2d = PlotConfig(
            title=f"FlipFlop Fixed Points ({config_name} - 2D PCA)",
            xlabel="PC 1", ylabel="PC 2", figsize=(10, 8),
            save_path=save_path_2d, show=False
        )
        plot_fixed_points_2d(unique_fps, hiddens_np, config=config_2d)
        print(f"\nSaved 2D plot to: {save_path_2d}")

        save_path_3d = f"flipflop_{config_name}_fixed_points_3d.png"
        config_3d = PlotConfig(
            title=f"FlipFlop Fixed Points ({config_name} - 3D PCA)",
            figsize=(12, 10), save_path=save_path_3d, show=False
        )
        plot_fixed_points_3d(
            unique_fps, hiddens_np, config=config_3d,
            plot_batch_idx=list(range(30)), plot_start_time=10
        )
        print(f"Saved 3D plot to: {save_path_3d}")

    print("\n--- Analysis complete ---")


if __name__ == "__main__":

    config_to_run = "3_bit"  # Specify the desired configuration here: "2_bit","3_bit","4_bit"
    # Use fixed seed
    seed_to_use = 42
    # Use random seed
    # seed_to_use = None

    print(f"\n--- Running configuration: {config_to_run} ---")
    if seed_to_use is None:
        main(config_name=config_to_run)
    else:
        main(config_name=config_to_run, seed=seed_to_use)

    print(f"\n--- Finished configuration: {config_to_run} ---")
