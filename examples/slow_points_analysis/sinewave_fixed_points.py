"""Sine wave generator task with fixed point analysis using BrainState.

This example trains an RNN to generate sine waves of different frequencies,
based on a constant input signal that specifies the target frequency.

This task, unlike the FlipFlop memory task, relies on unstable fixed points
(saddles) to create limit cycle oscillations. The fixed points are
conditionally dependent on the static input value.

Based on the task described in:
Sussillo, D., & Barak, O. (2013). Opening the black box: low-dimensional
dynamics in high-dimensional recurrent neural networks. Neural Computation.
"""

import random

import brainpy as bp
import brainpy.math as bm
import jax
import jax.numpy as jnp
import numpy as np

from canns.analyzer.visualization import PlotConfig
from canns.analyzer.slow_points import FixedPointFinder, plot_fixed_points_2d


def generate_sine_wave_data(n_trials=128, n_steps=100):
    t = np.linspace(0, 10 * np.pi, n_steps)
    sequences = np.array([np.sin(t + np.random.uniform(-np.pi, np.pi)) for _ in range(n_trials)])

    inputs = sequences[:, :-1]
    targets = sequences[:, 1:]

    return {
        "inputs": inputs[..., None].astype(np.float32),
        "targets": targets[..., None].astype(np.float32)
    }


class SineWaveRNN(bp.DynamicalSystem):
    """
    A simple RNN for sine wave prediction
    """

    def __init__(self, hidden_size=32, seed=0):
        super().__init__()
        self.input_size = 1
        self.hidden_size = hidden_size
        self.n_outputs = 1
        self.rnn_type = "tanh"  #

        key = jax.random.PRNGKey(seed)
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)

        # Adopting orthogonal initialization and high gain
        def orthogonal(key, shape, gain=1.0):
            a = jax.random.normal(key, shape)
            q, r = jnp.linalg.qr(a)
            q = q * jnp.sign(jnp.diag(r))
            return gain * q

        # Input weights use small random values
        self.w_ih = bm.Variable(
            jax.random.normal(k1, (self.input_size, self.hidden_size)) * 0.1
        )

        # Promote sustained oscillation
        self.w_hh = bm.Variable(orthogonal(k2, (self.hidden_size, self.hidden_size), gain=1.1))
        self.b_h = bm.Variable(jnp.zeros(self.hidden_size))
        self.w_out = bm.Variable(
            jax.random.normal(k3, (self.hidden_size, self.n_outputs)) * 0.1
        )
        self.b_out = bm.Variable(jnp.zeros(self.n_outputs))
        self.h0 = bm.Variable(jnp.zeros(self.hidden_size))

    def step(self, x_t, h):
        """Single RNN step (Tanh)."""
        h_next = jnp.tanh(
            x_t @ self.w_ih.value + h @ self.w_hh.value + self.b_h.value
        )
        return h_next

    def __call__(self, inputs, hidden=None):
        """Forward pass."""
        batch_size = inputs.shape[0]
        n_time = inputs.shape[1]

        if hidden is None:
            h = jnp.tile(self.h0.value, (batch_size, 1))
        else:
            h = hidden

        if n_time == 1:
            x_t = inputs[:, 0, :]
            h_next = self.step(x_t, h)
            y = h_next @ self.w_out.value + self.b_out.value
            return y[:, None, :], h_next

        def scan_fn(carry, x_t):
            h_prev = carry
            h_next = self.step(x_t, h_prev)
            y_t = h_next @ self.w_out.value + self.b_out.value
            return h_next, (y_t, h_next)

        inputs_transposed = inputs.transpose(1, 0, 2)
        _, (outputs_seq, hiddens_seq) = jax.lax.scan(scan_fn, h, inputs_transposed)
        outputs = outputs_seq.transpose(1, 0, 2)
        hiddens = hiddens_seq.transpose(1, 0, 2)

        return outputs, hiddens


def train_sine_wave_rnn(rnn, train_data, valid_data,
                        learning_rate=0.01,
                        batch_size=128,
                        max_epochs=200,
                        min_loss=1e-5,
                        print_every=10):
    print("\n" + "=" * 70)
    print("Training Sine Wave RNN (FlipFlop Style)")
    print("=" * 70)

    train_inputs = jnp.array(train_data['inputs'])
    train_targets = jnp.array(train_data['targets'])
    valid_inputs = jnp.array(valid_data['inputs'])
    valid_targets = jnp.array(valid_data['targets'])
    n_train = train_inputs.shape[0]
    n_batches = max(1, n_train // batch_size)

    # Get trainable variables from the model
    # Note: vars() returns keys like 'SineWaveRNN0.w_ih', we need just 'w_ih' for computation
    train_vars = {name: var for name, var in rnn.vars().items() if isinstance(var, bm.Variable)}
    # Create mapping between short names and full names
    name_mapping = {name.split('.')[-1]: name for name in train_vars.keys()}
    # Extract just the parameter name (after the last dot) for gradient computation
    params = {name.split('.')[-1]: var.value for name, var in train_vars.items()}

    # Initialize optimizer with train_vars parameter (modern brainpy API)
    optimizer = bp.optimizers.Adam(lr=learning_rate, train_vars=train_vars)

    @jax.jit
    def grad_step(params, batch_inputs, batch_targets):
        """Pure function to compute loss and gradients"""

        def forward_pass(p, inputs):
            batch_size = inputs.shape[0]
            h = jnp.tile(p['h0'], (batch_size, 1))

            def scan_fn(carry, x_t):
                h_prev = carry
                h_next = jnp.tanh(x_t @ p['w_ih'] + h_prev @ p['w_hh'] + p['b_h'])
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

    print("\nTraining complete!")
    return losses


if __name__ == '__main__':
    seed = np.random.randint(1, 10000)
    np.random.seed(seed)

    random.seed(seed)  #
    print(f"Running with global seed: {seed}")

    # Initialize the model
    model = SineWaveRNN(hidden_size=32, seed=seed)  #

    # Generate data
    train_data = generate_sine_wave_data(n_trials=128)
    valid_data = generate_sine_wave_data(n_trials=64)

    # Train the model
    train_sine_wave_rnn(
        model,
        train_data,
        valid_data,
        learning_rate=0.01,
        max_epochs=200,
        min_loss=1e-5
    )

    # Generate trajectories for analysis
    print("\nGenerating trajectories for Fixed Point analysis...")
    print("  Generating 64 trajectories for robust fixed point *finding*...")
    find_test_data = generate_sine_wave_data(n_trials=64)  #
    find_inputs_jax = jnp.array(find_test_data['inputs'])
    _, find_hiddens_jax = model(find_inputs_jax)
    find_hiddens_np = np.asarray(find_hiddens_jax)  # Shape (64, T, H)
    print(f"  Generated 'find' trajectories, shape: {find_hiddens_np.shape}")

    print("  Generating 1 trajectory for clean plotting")
    plot_test_data = generate_sine_wave_data(n_trials=1)
    plot_inputs_jax = jnp.array(plot_test_data['inputs'])
    _, plot_hiddens_jax = model(plot_inputs_jax)
    plot_hiddens_np = np.asarray(plot_hiddens_jax)  # Shape (1, T, H)


    def normalize_hiddens(h_arr, expected_T=None, name="hiddens"):
        # ensure numpy array
        h = np.asarray(h_arr)
        print(f"[DEBUG] raw {name} shape: {h.shape}")
        # if time-first (T, B, H) and expected_T known, transpose
        if h.ndim == 3 and expected_T is not None and h.shape[0] == expected_T and h.shape[1] != expected_T:
            print(f"[DEBUG] detected {name} likely (T,B,H) -> transposing to (B,T,H).")
            h = h.transpose(1, 0, 2)
        # if single-step (B, H), expand time axis
        if h.ndim == 2:
            print(f"[DEBUG] detected {name} shape (B,H) -> expanding to (B,1,H).")
            h = h[:, None, :]
        if h.ndim != 3:
            raise RuntimeError(f"[ERROR] normalized {name} must be 3D (B,T,H), got {h.shape}")
        if expected_T is not None and h.shape[1] != expected_T:
            print(f"[WARNING] {name} time-dim mismatch: expected T={expected_T}, got T={h.shape[1]}")
        print(f"[DEBUG] normalized {name} shape -> {h.shape}")
        return h


    expected_T = find_inputs_jax.shape[1] if 'find_inputs_jax' in locals() else None
    find_hiddens_np = normalize_hiddens(find_hiddens_jax, expected_T=expected_T, name="find_hiddens_np")

    expected_T_plot = plot_inputs_jax.shape[1] if 'plot_inputs_jax' in locals() else None
    plot_hiddens_np = normalize_hiddens(plot_hiddens_jax, expected_T=expected_T_plot, name="plot_hiddens_np")

    print(f"  Generated 'plot' trajectories, shape: {plot_hiddens_np.shape}")
    print("\n--- Fixed Point Analysis (Manual flipflop_fixed_points.py style) ---")

    finder = FixedPointFinder(
        model,
        method="joint",
        max_iters=2000,
        lr_init=0.02,
        tol_q=1e-5,
        final_q_threshold=1e-6,
        tol_unique=0.3,
        do_compute_jacobians=True,
        do_decompose_jacobians=True,
        verbose=True,
        super_verbose=True,
    )

    # n_inputs = model.input_size
    n_inputs = getattr(model, "input_size", 1)
    # constant_input = np.zeros((1, n_inputs), dtype=np.float32)
    constant_input = np.zeros((1, int(n_inputs)), dtype=np.float32)
    print(f"[DEBUG] constant_input shape={constant_input.shape}, dtype={constant_input.dtype}")
    print(f"[DEBUG] constant_input values:\n{constant_input}")
    print(f"[DEBUG] constant_input min={float(np.min(constant_input))}, max={float(np.max(constant_input))}")

    print(f"Searching for fixed points with {constant_input.shape} constant input...")

    unique_fps, all_fps = finder.find_fixed_points(
        state_traj=find_hiddens_np,
        inputs=constant_input,
        n_inits=1024,
        noise_scale=0.4
    )

    # Print and visualize
    print("\n--- Fixed Point Analysis Results ---")
    unique_fps.print_summary()  #

    if unique_fps.n > 0:
        print(f"\nDetailed Fixed Point Information (Top 10):")
        print(f"{'#':<4} {'q-value':<12} {'Stability':<12} {'Max |eig|':<12}")
        print("-" * 45)
        for i in range(min(10, unique_fps.n)):
            stability_str = "Stable" if unique_fps.is_stable[i] else "Unstable"
            max_eig = np.max(np.abs(unique_fps.eigval_J_xstar[i]))
            print(
                f"{i + 1:<4} {unique_fps.qstar[i]:<12.2e} {stability_str:<12} {max_eig:<12.4f}"
            )

        save_path_2d = "sine_wave_predictor_fixed_points_2d.png"
        config_2d = PlotConfig(
            title="Sine Wave Predictor Fixed Points (2D PCA)",
            xlabel="PC 1", ylabel="PC 2", figsize=(10, 8),
            save_path=save_path_2d, show=False
        )
        plot_fixed_points_2d(unique_fps, plot_hiddens_np, config=config_2d, plot_start_time=10)  # 忽略前 10 个时间步
        print(f"\nSaved 2D plot to: {save_path_2d}")

    print("\n--- Analysis complete ---")
