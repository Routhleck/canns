"""
STDP (Spike-Timing-Dependent Plasticity) for temporal pattern learning.

This example demonstrates:
- STDPTrainer with JIT compilation for spike-based learning
- Temporal credit assignment: pre→post timing strengthens weights (LTP)
- Reverse timing weakens weights (LTD)
- Learning spike sequences and temporal patterns
- Weight evolution visualization showing timing-dependent changes
"""

import brainpy.math as bm
import numpy as np
from matplotlib import pyplot as plt

from canns.models.brain_inspired import SpikingLayer
from canns.trainer import STDPTrainer

np.random.seed(42)
bm.random.seed(42)

# ========================================================================
# Generate temporal spike patterns
# ========================================================================
n_input = 20  # Number of input neurons
n_output = 5  # Number of output neurons
n_patterns = 100  # Number of spike patterns
n_timesteps = 50  # Length of each temporal pattern

print("Generating temporal spike patterns...")
print(f"Input neurons: {n_input}, Output neurons: {n_output}")
print(f"Patterns: {n_patterns}, Timesteps per pattern: {n_timesteps}")

# Create spike patterns with temporal structure
# Pattern 1: First 5 inputs spike early, should drive output neurons
# Pattern 2: Next 5 inputs spike slightly later
spike_patterns = []

for _ in range(n_patterns):
    pattern_sequence = []
    for t in range(n_timesteps):
        # Poisson spike generation with time-varying rates
        spikes = np.zeros(n_input)

        # Early input group (0-4) has high rate at t < 10
        if t < 10:
            spikes[0:5] = (np.random.rand(5) < 0.3).astype(np.float32)

        # Middle input group (5-9) has high rate at 10 < t < 20
        if 10 < t < 20:
            spikes[5:10] = (np.random.rand(5) < 0.3).astype(np.float32)

        # Late input group (10-14) has high rate at 20 < t < 30
        if 20 < t < 30:
            spikes[10:15] = (np.random.rand(5) < 0.3).astype(np.float32)

        # Background noise
        spikes[15:] = (np.random.rand(5) < 0.05).astype(np.float32)

        pattern_sequence.append(spikes)

    spike_patterns.append(pattern_sequence)

print(f"Generated {len(spike_patterns)} temporal spike patterns")

# ========================================================================
# Create model and trainer
# ========================================================================
model = SpikingLayer(
    input_size=n_input,
    output_size=n_output,
    threshold=0.8,  # Lower threshold for more spiking
    v_reset=0.0,
    leak=0.95,  # High leak for temporal integration
    trace_decay=0.90,  # Trace decay determines STDP time window
    dt=1.0,
)

# Store initial weights for comparison
initial_weights = model.W.value.copy()

# Create STDP trainer with JIT compilation
# A_minus slightly > A_plus for stable competition
trainer = STDPTrainer(
    model,
    learning_rate=0.02,
    A_plus=0.005,
    A_minus=0.00525,
    w_min=0.0,
    w_max=1.0,
    compiled=True,
)

print("\nTraining with STDP (JIT compiled)...")
print(f"Learning rate: {trainer.learning_rate}")
print(f"A_plus (LTP): {trainer.A_plus}, A_minus (LTD): {trainer.A_minus}")

# ========================================================================
# Train on temporal patterns
# ========================================================================
n_epochs = 20
weight_history = [initial_weights.copy()]
spike_counts_history = []

for epoch in range(n_epochs):
    # Reset model state at start of each epoch
    model.reset_state()

    epoch_spike_counts = []

    # Train on all patterns
    for pattern_seq in spike_patterns:
        # Train on each timestep of the pattern
        trainer.train(pattern_seq)

        # Count total output spikes for this pattern
        # (Collect spikes during the pattern presentation)
        pattern_spikes = []
        model.reset_state()
        for spike_input in pattern_seq:
            spike_output = model.forward(spike_input)
            pattern_spikes.append(spike_output)
        epoch_spike_counts.append(np.sum(pattern_spikes))

    # Track metrics
    spike_counts_history.append(np.mean(epoch_spike_counts))

    if (epoch + 1) % 5 == 0:
        weight_history.append(model.W.value.copy())
        print(f"Epoch {epoch + 1}/{n_epochs}: Avg spikes/pattern = {spike_counts_history[-1]:.2f}")

print("\nTraining complete!")

# Final weights
final_weights = model.W.value

# ========================================================================
# Analyze learned weights
# ========================================================================
print("\n" + "=" * 70)
print("Weight Change Analysis")
print("=" * 70)

# Compare initial vs final weights for each output neuron
weight_change = final_weights - initial_weights

print("\nWeight changes by input group:")
for i in range(n_output):
    early_change = np.mean(weight_change[i, 0:5])
    middle_change = np.mean(weight_change[i, 5:10])
    late_change = np.mean(weight_change[i, 10:15])
    noise_change = np.mean(weight_change[i, 15:])

    print(f"Neuron {i}: Early={early_change:+.4f}, Middle={middle_change:+.4f}, "
          f"Late={late_change:+.4f}, Noise={noise_change:+.4f}")

# ========================================================================
# Test on a single pattern to visualize spike timing
# ========================================================================
print("\n" + "=" * 70)
print("Testing spike timing on sample pattern")
print("=" * 70)

model.reset_state()
test_pattern = spike_patterns[0]
input_spikes = []
output_spikes = []

for t, spike_input in enumerate(test_pattern):
    spike_output = model.forward(spike_input)
    input_spikes.append(spike_input)
    output_spikes.append(spike_output)

input_spikes = np.array(input_spikes)  # (n_timesteps, n_input)
output_spikes = np.array(output_spikes)  # (n_timesteps, n_output)

print(f"Input spike count: {np.sum(input_spikes)}")
print(f"Output spike count: {np.sum(output_spikes)}")

# ========================================================================
# Visualize results
# ========================================================================
fig = plt.figure(figsize=(15, 10))

# Create grid for subplots
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Plot 1: Initial weights (heatmap)
ax1 = fig.add_subplot(gs[0, 0])
im1 = ax1.imshow(initial_weights, aspect="auto", cmap="RdBu_r", vmin=0, vmax=1)
ax1.set_xlabel("Input Neuron")
ax1.set_ylabel("Output Neuron")
ax1.set_title("Initial Weights")
plt.colorbar(im1, ax=ax1)

# Plot 2: Final weights (heatmap)
ax2 = fig.add_subplot(gs[0, 1])
im2 = ax2.imshow(final_weights, aspect="auto", cmap="RdBu_r", vmin=0, vmax=1)
ax2.set_xlabel("Input Neuron")
ax2.set_ylabel("Output Neuron")
ax2.set_title("Final Weights (After STDP)")
plt.colorbar(im2, ax=ax2)

# Plot 3: Weight change (heatmap)
ax3 = fig.add_subplot(gs[0, 2])
im3 = ax3.imshow(weight_change, aspect="auto", cmap="RdBu_r",
                 vmin=-np.max(np.abs(weight_change)), vmax=np.max(np.abs(weight_change)))
ax3.set_xlabel("Input Neuron")
ax3.set_ylabel("Output Neuron")
ax3.set_title("Weight Change (ΔW)")
plt.colorbar(im3, ax=ax3)

# Add vertical lines to show input groups
for ax in [ax1, ax2, ax3]:
    ax.axvline(x=4.5, color='white', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(x=9.5, color='white', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(x=14.5, color='white', linestyle='--', alpha=0.5, linewidth=1)

# Plot 4: Input spike raster (test pattern)
ax4 = fig.add_subplot(gs[1, :])
for neuron_idx in range(n_input):
    spike_times = np.where(input_spikes[:, neuron_idx] > 0)[0]
    ax4.scatter(spike_times, [neuron_idx] * len(spike_times), marker='|', s=100, c='black')

ax4.set_xlabel("Time Step")
ax4.set_ylabel("Input Neuron")
ax4.set_title("Input Spike Raster (Test Pattern)")
ax4.set_ylim([-0.5, n_input - 0.5])
ax4.axhline(y=4.5, color='red', linestyle='--', alpha=0.3, linewidth=1)
ax4.axhline(y=9.5, color='green', linestyle='--', alpha=0.3, linewidth=1)
ax4.axhline(y=14.5, color='blue', linestyle='--', alpha=0.3, linewidth=1)
ax4.grid(True, alpha=0.3, axis='x')

# Plot 5: Output spike raster (test pattern)
ax5 = fig.add_subplot(gs[2, 0])
for neuron_idx in range(n_output):
    spike_times = np.where(output_spikes[:, neuron_idx] > 0)[0]
    ax5.scatter(spike_times, [neuron_idx] * len(spike_times), marker='|', s=150, c='red')

ax5.set_xlabel("Time Step")
ax5.set_ylabel("Output Neuron")
ax5.set_title("Output Spike Raster")
ax5.set_ylim([-0.5, n_output - 0.5])
ax5.grid(True, alpha=0.3)

# Plot 6: Spike count evolution
ax6 = fig.add_subplot(gs[2, 1])
ax6.plot(range(1, n_epochs + 1), spike_counts_history, marker='o', linewidth=2)
ax6.set_xlabel("Epoch")
ax6.set_ylabel("Avg Spikes per Pattern")
ax6.set_title("Output Spike Activity Over Training")
ax6.grid(True, alpha=0.3)

# Plot 7: Weight evolution for selected neurons
ax7 = fig.add_subplot(gs[2, 2])
# Track weights from first input group (early spikers) to first output neuron
epochs_tracked = [0] + list(range(5, n_epochs + 1, 5))
early_weights = [w[0, 0:5].mean() for w in weight_history]
middle_weights = [w[0, 5:10].mean() for w in weight_history]
late_weights = [w[0, 10:15].mean() for w in weight_history]

ax7.plot(epochs_tracked, early_weights, label="Early inputs (0-4)", marker='o', linewidth=2)
ax7.plot(epochs_tracked, middle_weights, label="Middle inputs (5-9)", marker='s', linewidth=2)
ax7.plot(epochs_tracked, late_weights, label="Late inputs (10-14)", marker='^', linewidth=2)
ax7.set_xlabel("Epoch")
ax7.set_ylabel("Average Weight (to Neuron 0)")
ax7.set_title("Weight Evolution by Input Group")
ax7.legend()
ax7.grid(True, alpha=0.3)

plt.savefig("stdp_temporal_learning.png", dpi=150, bbox_inches='tight')
print("\nPlot saved as 'stdp_temporal_learning.png'")
plt.show()

# ========================================================================
# Summary
# ========================================================================
print("\n" + "=" * 70)
print("SUMMARY: STDP Temporal Learning")
print("=" * 70)
print("""
STDP Key Principles:
1. Causality: Pre-synaptic spikes before post-synaptic → LTP (strengthen)
2. Anti-causality: Post-synaptic spikes before pre-synaptic → LTD (weaken)
3. Temporal window: Spike traces (decay = 0.90) define timing sensitivity

Expected Results:
- Early-spiking inputs (0-4) should develop strong weights
  → They consistently spike before outputs can spike
- Middle inputs (5-9) should have moderate weight changes
  → Mixed temporal relationship with output spikes
- Late inputs (10-14) may experience LTD
  → Often spike after outputs have already spiked
- Noise inputs (15-19) should remain weak
  → No consistent temporal correlation

STDP enables:
- Temporal pattern learning without supervised labels
- Self-organization based on spike timing statistics
- Biologically plausible credit assignment across time
""")
