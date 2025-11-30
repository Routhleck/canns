"""
Sine wave generator task with fixed point analysis using BrainState.

This example trains an RNN to generate sine waves of different frequencies,
based on a constant input signal that specifies the target frequency.

This task, unlike the FlipFlop memory task, relies on unstable fixed points
(saddles) to create limit cycle oscillations. The fixed points are
conditionally dependent on the static input value.

Based on the task described in:
Sussillo, D., & Barak, O. (2013). Opening the black box: low-dimensional
dynamics in high-dimensional recurrent neural networks. Neural Computation.
"""


import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import jax
import jax.numpy as jnp
import brainstate as bst
import braintools as bts


class Config:
    # Task Parameters
    N_FREQS = 51
    HIDDEN_SIZE = 200

    # Simulation Parameters
    DT = 0.01
    N_STEPS = 400

    # Dynamics (Physical Inertia)
    # Tau = 0.1s => alpha = dt/tau = 0.1
    # State updates by 10% each step, providing necessary inertia.
    TAU = 0.1

    # Training Parameters
    BATCH_SIZE = 64
    STEPS_PER_EPOCH = 50
    MAX_EPOCHS = 1500

    # Learning Rate Schedule
    DECAY_STEP_1 = 800 * STEPS_PER_EPOCH
    DECAY_STEP_2 = 1200 * STEPS_PER_EPOCH

    VALID_SIZE = 1024
    SEED = 1234


def get_random_batch(batch_size, n_steps=Config.N_STEPS, n_freqs=Config.N_FREQS):
    """
    Generate training data
    """
    # 1. Randomly select frequency indices (1 to 51)
    freq_indices = np.random.randint(0, n_freqs, size=batch_size)

    # 2. Input Mapping
    # Input = j/51 + 0.25
    inputs_u = (freq_indices / float(n_freqs)) + 0.25

    # 3. Target Frequencies Mapping
    # Original: 0.1-0.6 rad/s. Scaled to 1.0-6.0 rad/s for efficient training.
    min_omega, max_omega = 1.0, 6.0
    omegas = min_omega + (freq_indices / float(n_freqs - 1)) * (max_omega - min_omega)

    # 4. Generate Time Vector
    t = np.arange(0, n_steps * Config.DT, Config.DT, dtype=np.float32)

    batch_x = []
    batch_y = []

    for i in range(batch_size):
        u = inputs_u[i]
        omega = omegas[i]

        # Target: Sine wave
        target = np.sin(omega * t)

        # Input: Static constant value
        inp = np.full((n_steps, 1), u, dtype=np.float32)

        batch_x.append(inp)
        batch_y.append(target[:, None])

    return jnp.array(batch_x), jnp.array(batch_y)


class SineWaveCTRNN(bst.nn.Module):
    def __init__(self, hidden_size, out_size=1, seed=0, dt=Config.DT, tau=Config.TAU):
        super().__init__()
        self.hidden_size = hidden_size
        self.out_size = out_size

        # Inertia coefficient alpha
        self.alpha = dt / tau

        key = jax.random.PRNGKey(seed)
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        std = 1.0 / np.sqrt(hidden_size)

        # Weight Initialization
        self.w_ih = bst.ParamState(jax.random.normal(k1, (1, hidden_size)) * std)
        self.w_hh = bst.ParamState(jax.random.normal(k2, (hidden_size, hidden_size)) * std)
        self.b_h = bst.ParamState(jnp.zeros(hidden_size))

        self.w_out = bst.ParamState(jax.random.normal(k4, (hidden_size, out_size)) * std)
        self.b_out = bst.ParamState(jnp.zeros(out_size))

        # Initial State (Membrane Potential x)
        self.x0 = bst.ParamState(jnp.zeros(hidden_size))

    def cell_step(self, u_t, x_prev):
        """
        Euler Integration Step
        Equation 6.4: tau * x_dot = -x + J*r + B*u + b
        """
        # 1. Compute firing rate r = tanh(x)
        r_prev = jnp.tanh(x_prev)

        # 2. Compute Input Current (Input + Recurrent + Bias)
        current_in = u_t @ self.w_ih.value + r_prev @ self.w_hh.value + self.b_h.value

        # 3. Update Membrane Potential x (Euler Method)
        # x_new = (1 - alpha) * x_old + alpha * current_in
        x_next = (1 - self.alpha) * x_prev + self.alpha * current_in

        return x_next

    def __call__(self, inputs, hidden=None):
        batch_size = inputs.shape[0]
        n_time = inputs.shape[1]

        if hidden is None:
            # 'hidden' here refers to membrane potential x
            x = jnp.tile(self.x0.value, (batch_size, 1))
        else:
            x = hidden

        # Single step execution (for FixedPointFinder)
        if n_time == 1:
            u_t = inputs[:, 0, :]
            x_next = self.cell_step(u_t, x)
            # Output is based on firing rate r
            r_next = jnp.tanh(x_next)
            y = r_next @ self.w_out.value + self.b_out.value
            return y[:, None, :], x_next

        # Sequence Scan
        def scan_fn(carry, u_t):
            x_prev = carry
            x_next = self.cell_step(u_t, x_prev)

            r_next = jnp.tanh(x_next)
            y_t = r_next @ self.w_out.value + self.b_out.value

            return x_next, (y_t, x_next)

        inputs_T = inputs.transpose(1, 0, 2)
        _, (outputs_T, hiddens_T) = jax.lax.scan(scan_fn, x, inputs_T)

        return outputs_T.transpose(1, 0, 2), hiddens_T.transpose(1, 0, 2)


def train(model):
    print(f"Setting up training (CTRNN Reproduction)...")
    print(f"  Frequencies: {Config.N_FREQS}")
    print(f"  Network: N={Config.HIDDEN_SIZE}, Tau={Config.TAU}, dt={Config.DT}")

    valid_x, valid_y = get_random_batch(Config.VALID_SIZE)

    # Learning Rate Schedule
    lr_schedule = bts.optim.PiecewiseConstantSchedule(
        boundaries=[Config.DECAY_STEP_1, Config.DECAY_STEP_2],
        values=[1e-3, 1e-4, 1e-5]
    )

    optimizer = bts.optim.Adam(lr=lr_schedule)
    optimizer.register_trainable_weights(bst.graph.states(model, bst.ParamState))

    def loss_fn(inputs, targets):
        preds, _ = model(inputs)
        return jnp.mean((preds - targets) ** 2)

    @bst.transform.jit
    def train_step(batch_x, batch_y):
        def loss_wrapper(in_x, in_y):
            l = loss_fn(in_x, in_y)
            return l, l

        grad_fn = bst.transform.grad(
            loss_wrapper,
            grad_states=bst.graph.states(model, bst.ParamState),
            has_aux=True
        )

        grads, loss = grad_fn(batch_x, batch_y)
        optimizer.update(grads)
        return loss

    @bst.transform.jit
    def valid_step(batch_x, batch_y):
        return loss_fn(batch_x, batch_y)

    # Training Loop
    start_time = time.time()
    global_step = 0

    for epoch in range(Config.MAX_EPOCHS):
        epoch_loss = 0.0
        for _ in range(Config.STEPS_PER_EPOCH):
            bx, by = get_random_batch(Config.BATCH_SIZE)
            loss_val = train_step(bx, by)
            epoch_loss += float(loss_val)
            global_step += 1

        epoch_loss /= Config.STEPS_PER_EPOCH

        if epoch % 50 == 0:
            val_loss = valid_step(valid_x, valid_y)
            print(f"Epoch {epoch:4d} | Train: {epoch_loss:.6f} | Valid: {val_loss:.6f}")

            if val_loss < 1e-4:
                print("Converged! Stopping early.")
                break

    print(f"Training finished in {time.time() - start_time:.2f}s")


def analyze_and_plot(model):
    print("\n--- Starting Fixed Point Analysis ---")
    try:
        from canns.analyzer.slow_points.finder import FixedPointFinder
    except ImportError:
        print("Analysis library 'canns' not found. Skipping.")
        return

    # 1. Generate inputs for all 51 frequencies sequentially
    print("Generating trajectories for all 51 frequencies...")
    inputs_list = []

    for i in range(Config.N_FREQS):
        u_val = (i / float(Config.N_FREQS)) + 0.25
        # Shape: (1, N_STEPS, 1)
        inp = np.full((1, Config.N_STEPS, 1), u_val, dtype=np.float32)
        inputs_list.append(inp)

    inputs_all = np.concatenate(inputs_list, axis=0)  # (51, 400, 1)

    # 2. Run model to get trajectories (Membrane potential 'x')
    _, traj_states = model(jnp.array(inputs_all))
    traj_states = np.array(traj_states)  # (51, 400, 200)

    # 3. Initialize Finder
    # Input-Dependent Fixed Points: Run Finder separately for each frequency condition
    finder = FixedPointFinder(
        model,
        method="joint",
        max_iters=50000,
        tol_q=1e-6,
        verbose=True,
        super_verbose=True,
        n_iters_per_print_update=500,
        lr_init=0.1,
        lr_patience=1000,
        lr_factor=0.8,
        final_q_threshold=1e-6,
        tol_unique=0.5
    )

    all_fps_list = []
    valid_freq_count = 0

    print(f"Finding fixed points for each frequency condition...")

    for i in range(Config.N_FREQS):
        # Use current trajectory for initialization
        curr_traj = traj_states[i]  # (400, 200)
        curr_u = inputs_all[i, 0, :]  # (1,)

        # Adapt input shape for Finder: (1, 1)
        u_for_finder = curr_u[None, :]
        center_guess = np.mean(curr_traj, axis=0, keepdims=True)  # (1, 200)
        fake_traj_for_init = np.tile(center_guess, (1, 10, 1))

        # Run optimization
        unique_fps, _ = finder.find_fixed_points(
            state_traj=fake_traj_for_init,
            inputs=u_for_finder,
            n_inits=32,
            noise_scale=0.1,
            valid_bxt=None
        )

        if unique_fps.n > 0:
            # Select the slowest point (min qstar)
            best_idx = np.argmin(unique_fps.qstar)
            best_fp = unique_fps.xstar[best_idx]
            best_q = unique_fps.qstar[best_idx]

            all_fps_list.append(best_fp)
            valid_freq_count += 1

            if i % 10 == 0:
                print(f"Freq {i}/{Config.N_FREQS}: Found FP with q={best_q:.2e}")
        else:
            if i % 10 == 0:
                print(f"Freq {i}/{Config.N_FREQS}: No FP found.")

    print(f"Found fixed points for {valid_freq_count} / {Config.N_FREQS} frequencies.")

    if not all_fps_list:
        return

    # 4. Visualization (PCA)
    print("Plotting Manifold...")
    fp_array = np.array(all_fps_list)

    # Flatten trajectories for PCA training
    traj_flat = traj_states.reshape(-1, Config.HIDDEN_SIZE)

    # Fit PCA on mixed data
    combined_data = np.concatenate([traj_flat, fp_array], axis=0)
    pca = PCA(n_components=3)
    pca.fit(combined_data)

    traj_pca = pca.transform(traj_states.reshape(-1, Config.HIDDEN_SIZE)).reshape(Config.N_FREQS, Config.N_STEPS, 3)
    fp_pca = pca.transform(fp_array)

    # Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter Fixed Points (Red X)
    ax.scatter(fp_pca[:, 0], fp_pca[:, 1], fp_pca[:, 2],
               color='red', marker='x', s=50, label='Fixed Points', alpha=0.9, zorder=1)

    # Plot Trajectories (Blue Lines)
    for i in range(0, Config.N_FREQS, 5):
        ax.plot(traj_pca[i, 50:, 0], traj_pca[i, 50:, 1], traj_pca[i, 50:, 2],
                color='blue', alpha=0.3, lw=0.8, zorder=1)

    ax.set_title(f"BrainState CTRNN: Sine Generator Manifold (N={Config.N_FREQS})")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.legend()

    plt.savefig("sine_ctrnn_repro.png", dpi=150)
    print("Saved 'sine_ctrnn_repro.png'. Analysis Complete.")


def main():
    np.random.seed(Config.SEED)
    random.seed(Config.SEED)

    # Initialize Model
    model = SineWaveCTRNN(Config.HIDDEN_SIZE, seed=Config.SEED)

    # Train
    train(model)

    # Analyze
    analyze_and_plot(model)


if __name__ == '__main__':
    main()