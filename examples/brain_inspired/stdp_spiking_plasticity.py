"""
Spike-Timing-Dependent Plasticity (STDP) with LIF neurons.

This example demonstrates:
- STDPTrainer with LIF spiking network
- Temporal learning based on spike timing
- STDP window visualization
"""

import numpy as np
from matplotlib import pyplot as plt

from canns.models.brain_inspired import LIFSpikingNetwork
from canns.trainer import STDPTrainer

np.random.seed(42)

# Create LIF spiking network
n_neurons = 10
model = LIFSpikingNetwork(
    num_neurons=n_neurons,
    tau_m=20.0,  # Membrane time constant
    tau_syn=5.0,  # Synaptic time constant
    v_rest=-70.0,
    v_reset=-70.0,
    v_thresh=-55.0,
    dt=1.0,
)
model.init_state()

# Initialize with small random weights
key = np.random.RandomState(42)
model.W.value = key.randn(n_neurons, n_neurons).astype(np.float32) * 0.1
# Zero diagonal (no self-connections)
np.fill_diagonal(model.W.value, 0)

print(f"Created LIF network with {n_neurons} neurons")
print(f"Initial weight range: [{model.W.value.min():.3f}, {model.W.value.max():.3f}]")

# Create STDP trainer
trainer = STDPTrainer(
    model,
    a_plus=0.01,  # LTP amplitude
    a_minus=0.01,  # LTD amplitude
    tau_plus=20.0,  # LTP time constant
    tau_minus=20.0,  # LTD time constant
)

# Generate spike trains (input stimuli)
n_patterns = 5
n_steps_per_pattern = 100

print(f"\nGenerating {n_patterns} spike train patterns")

spike_patterns = []
for p in range(n_patterns):
    # Create pattern with temporal structure
    # Neurons fire in sequence with some jitter
    pattern_steps = []

    for step in range(n_steps_per_pattern):
        # Create sparse spike pattern
        spikes = np.zeros(n_neurons, dtype=np.float32)

        # Add sequential activation
        active_neuron = (step // 10) % n_neurons
        if np.random.random() < 0.3:  # Probabilistic spiking
            spikes[active_neuron] = 1.0

        # Add random background activity
        if np.random.random() < 0.05:
            random_neuron = np.random.randint(n_neurons)
            spikes[random_neuron] = 1.0

        pattern_steps.append(spikes)

    spike_patterns.append(pattern_steps)

# Record initial weights
W_initial = model.W.value.copy()

# Training: Run simulation with STDP
print("\nTraining with STDP...")

# Note: For this example, we manually simulate and apply STDP
# since STDP requires time-dependent dynamics
n_epochs = 3

for epoch in range(n_epochs):
    print(f"Epoch {epoch+1}/{n_epochs}")

    for pattern_idx, pattern in enumerate(spike_patterns):
        # Reset network state
        model.init_state()

        # Simulate pattern
        for step, spike_input in enumerate(pattern):
            model.set_input(spike_input)
            model.update(0.0)

            # Apply STDP update
            W_current = model.W.value
            W_updated = trainer._apply_stdp_update(W_current)
            model.W.value = W_updated

# Record final weights
W_final = model.W.value.copy()

print("\nTraining complete!")
print(f"Final weight range: [{W_final.min():.3f}, {W_final.max():.3f}]")

# Visualize results
fig = plt.figure(figsize=(15, 10))

# Plot 1: Weight change
ax1 = plt.subplot(2, 3, 1)
W_change = W_final - W_initial
im = ax1.imshow(W_change, cmap="RdBu_r", aspect="auto")
ax1.set_xlabel("Presynaptic Neuron")
ax1.set_ylabel("Postsynaptic Neuron")
ax1.set_title("Weight Changes (STDP)")
plt.colorbar(im, ax=ax1, label="ΔW")

# Plot 2: Initial weights
ax2 = plt.subplot(2, 3, 2)
im = ax2.imshow(W_initial, cmap="viridis", aspect="auto")
ax2.set_xlabel("Presynaptic Neuron")
ax2.set_ylabel("Postsynaptic Neuron")
ax2.set_title("Initial Weights")
plt.colorbar(im, ax=ax2)

# Plot 3: Final weights
ax3 = plt.subplot(2, 3, 3)
im = ax3.imshow(W_final, cmap="viridis", aspect="auto")
ax3.set_xlabel("Presynaptic Neuron")
ax3.set_ylabel("Postsynaptic Neuron")
ax3.set_title("Final Weights (After STDP)")
plt.colorbar(im, ax=ax3)

# Plot 4: STDP window
ax4 = plt.subplot(2, 3, 4)
dt_range = np.linspace(-50, 50, 100)
stdp_values = []

for dt in dt_range:
    if dt > 0:
        # LTP
        dw = trainer.a_plus * np.exp(-dt / trainer.tau_plus)
    else:
        # LTD
        dw = -trainer.a_minus * np.exp(dt / trainer.tau_minus)
    stdp_values.append(dw)

ax4.plot(dt_range, stdp_values, linewidth=2)
ax4.axhline(y=0, color="k", linestyle="--", alpha=0.5)
ax4.axvline(x=0, color="k", linestyle="--", alpha=0.5)
ax4.set_xlabel("Δt = t_post - t_pre (ms)")
ax4.set_ylabel("ΔW")
ax4.set_title("STDP Timing Window")
ax4.grid(True, alpha=0.3)
ax4.fill_between(dt_range, 0, stdp_values, where=np.array(stdp_values) > 0, alpha=0.3, color="g", label="LTP")
ax4.fill_between(dt_range, 0, stdp_values, where=np.array(stdp_values) < 0, alpha=0.3, color="r", label="LTD")
ax4.legend()

# Plot 5: Weight distribution
ax5 = plt.subplot(2, 3, 5)
ax5.hist(W_initial.flatten(), bins=30, alpha=0.5, label="Initial", edgecolor="black")
ax5.hist(W_final.flatten(), bins=30, alpha=0.5, label="Final", edgecolor="black")
ax5.set_xlabel("Weight Value")
ax5.set_ylabel("Frequency")
ax5.set_title("Weight Distribution")
ax5.legend()
ax5.grid(True, alpha=0.3, axis="y")

# Plot 6: Example spike raster for one pattern
ax6 = plt.subplot(2, 3, 6)
pattern = spike_patterns[0]
spike_times = []
neuron_ids = []

for t, spikes in enumerate(pattern):
    spiking_neurons = np.where(spikes > 0)[0]
    for neuron_id in spiking_neurons:
        spike_times.append(t)
        neuron_ids.append(neuron_id)

ax6.scatter(spike_times, neuron_ids, s=10, c="black", marker="|")
ax6.set_xlabel("Time (ms)")
ax6.set_ylabel("Neuron ID")
ax6.set_title("Example Spike Raster (Pattern 1)")
ax6.set_ylim([-0.5, n_neurons - 0.5])
ax6.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("stdp_spiking_plasticity.png")
print("\nPlot saved as 'stdp_spiking_plasticity.png'")
plt.show()

# Analyze weight changes
print("\n" + "=" * 60)
print("Weight Change Analysis")
print("=" * 60)

# Compute statistics
potentiated = W_change > 0.001
depressed = W_change < -0.001
unchanged = np.abs(W_change) <= 0.001

print(f"Synapses potentiated (LTP): {potentiated.sum()} ({potentiated.sum()/W_change.size:.1%})")
print(f"Synapses depressed (LTD): {depressed.sum()} ({depressed.sum()/W_change.size:.1%})")
print(f"Synapses unchanged: {unchanged.sum()} ({unchanged.sum()/W_change.size:.1%})")

print(f"\nMean weight change: {W_change.mean():.5f}")
print(f"Std weight change: {W_change.std():.5f}")
print(f"Max potentiation: {W_change.max():.5f}")
print(f"Max depression: {W_change.min():.5f}")

# Find strongest potentiated and depressed connections
flat_change = W_change.flatten()
flat_indices = np.arange(len(flat_change))

# Top 5 potentiated
top_potentiated_idx = np.argsort(flat_change)[-5:][::-1]
print("\nTop 5 potentiated connections:")
for idx in top_potentiated_idx:
    i, j = np.unravel_index(idx, W_change.shape)
    print(f"  {i} -> {j}: ΔW = {W_change[i, j]:.5f}")

# Top 5 depressed
top_depressed_idx = np.argsort(flat_change)[:5]
print("\nTop 5 depressed connections:")
for idx in top_depressed_idx:
    i, j = np.unravel_index(idx, W_change.shape)
    print(f"  {i} -> {j}: ΔW = {W_change[i, j]:.5f}")

# Plot detailed weight change analysis
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Histogram of weight changes
ax = axes[0]
ax.hist(W_change.flatten(), bins=50, edgecolor="black")
ax.axvline(x=0, color="r", linestyle="--", linewidth=2, label="No change")
ax.set_xlabel("Weight Change (ΔW)")
ax.set_ylabel("Frequency")
ax.set_title("Distribution of Weight Changes")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")

# Scatter: Initial weight vs change
ax = axes[1]
# Remove diagonal for clarity
mask = ~np.eye(n_neurons, dtype=bool)
ax.scatter(W_initial[mask], W_change[mask], alpha=0.5, s=20)
ax.axhline(y=0, color="r", linestyle="--", alpha=0.5)
ax.set_xlabel("Initial Weight")
ax.set_ylabel("Weight Change (ΔW)")
ax.set_title("Initial Weight vs. Change")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("stdp_weight_analysis.png")
print("\nWeight analysis plot saved as 'stdp_weight_analysis.png'")
plt.show()
