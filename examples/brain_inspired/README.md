# Brain-Inspired Learning Algorithms Examples

This directory demonstrates **classic brain-inspired learning algorithms** implemented as trainers in CANNs. The focus
is on the algorithms themselves, not model complexity.

## Philosophy

- **Algorithms over models**: Examples emphasize classic learning rules
- **Simplicity**: Use simple LinearLayer or Hopfield networks to showcase algorithms
- **Clarity**: Each example clearly demonstrates one algorithm's behavior

## Classic Algorithms

### Hebbian Learning

#### `hopfield_train.py`

Basic Hopfield network training on real images.

- **Trainer**: `HebbianTrainer`
- **Model**: `AmariHopfieldNetwork`
- **Features**: Pattern storage, corruption recovery

#### `hopfield_train_1d.py`

1D pattern storage and retrieval.

- **Trainer**: `HebbianTrainer`
- **Model**: `AmariHopfieldNetwork`
- **Features**: 1D binary patterns, visualization

#### `hopfield_train_mnist.py`

MNIST digit storage using Hopfield networks.

- **Trainer**: `HebbianTrainer`
- **Model**: `AmariHopfieldNetwork`
- **Features**: High-dimensional patterns

#### `hopfield_hebbian_vs_antihebbian.py`

Comparison of Hebbian vs Anti-Hebbian learning.

- **Trainers**: `HebbianTrainer`, `AntiHebbianTrainer`
- **Model**: `AmariHopfieldNetwork`
- **Features**: Side-by-side comparison, decorrelation effects

### Energy-Based Learning with Diagnostics

#### `hopfield_energy_diagnostics.py`

Advanced Hopfield network with comprehensive diagnostics.

- **Trainer**: `HebbianTrainer`
- **Analyzer**: `HopfieldAnalyzer` (from `canns.analyzer.brain_inspired`)
- **Model**: `AmariHopfieldNetwork`
- **Features**: Energy landscape, capacity analysis (N/4ln(N)), overlap metrics, recall quality diagnostics

This example demonstrates the **separation of concerns** design:

- **Trainer** (`HebbianTrainer`): Performs weight updates using Hebbian learning
- **Analyzer** (`HopfieldAnalyzer`): Provides diagnostics and analysis tools for Hopfield networks

The analyzer offers:

- `get_statistics()`: Capacity estimation, energy statistics, usage metrics
- `analyze_recall()`: Quality metrics for pattern retrieval
- `compute_overlap()`: Pattern similarity measurement
- `compute_energy()`: Energy calculation for any pattern

### Oja's Rule (Normalized Hebbian)

#### `oja_pca_extraction.py` - Principal Component Analysis through Unsupervised Learning

This example demonstrates how **Oja's Rule** naturally extracts principal components from high-dimensional data through
a biologically plausible learning mechanism.

**What it does:**

- Creates synthetic 50-dimensional data with 3 known principal components
- Trains a 3-neuron linear layer using Oja's normalized Hebbian learning
- Compares learned components with sklearn's PCA (ground truth)

**Why Oja's Rule?**

Pure Hebbian learning (`ΔW = η·y·x^T`) has a critical flaw: weights grow **unbounded** because co-activation always
strengthens connections. Oja's Rule adds a **normalization term** that prevents this:

```
ΔW = η·(y·x^T - y²·W)
      ↑          ↑
   Hebbian  Normalization (weight decay proportional to output²)
```

**The normalization term `y²·W`** acts as a forgetting mechanism that:

1. Prevents weight explosion
2. Naturally normalizes weight vectors to unit length
3. Makes neurons compete to capture different variance directions

**Expected Results:**

1. **Weight Norm Convergence** (top-left plot):
    - All weight norms should converge to **~1.0** (unit norm)
    - This happens automatically due to the normalization term
    - Without normalization, weights would grow exponentially

2. **Variance Explained** (top-right plot):
    - Each neuron captures **approximately equal variance** (~4.6)
    - Total variance ≈ 3 × 4.6 = 13.8 (out of total dataset variance ~50)
    - The 3 neurons collectively capture the 3 principal components

3. **Weight Matrix Structure** (bottom-left plot):
    - **First 30 dimensions show strong patterns** (the 3 structured components)
    - **Last 20 dimensions are near-zero** (noise is ignored)
    - Each row (neuron) learns to respond to a different component

4. **Alignment with PCA** (bottom-right plot):
    - Cosine similarity between Oja and sklearn PCA should be **>0.95**
    - This validates that Oja's Rule converges to true principal components
    - Small differences arise from: initialization, learning rate, finite epochs

**Why This Matters:**

Oja's Rule shows that **complex statistical operations** (PCA) can emerge from **simple local learning rules**. This is
how biological neural networks might perform dimensionality reduction without computing covariance matrices or
eigendecomposition.

**Parameters to Experiment:**

- `learning_rate`: Higher → faster convergence, but may overshoot
- `n_epochs`: More epochs → better convergence to true PCs
- `n_components`: Can extract more or fewer principal components

**Trainer**: `OjaTrainer`
**Model**: `LinearLayer`
**Output**: `oja_pca_extraction.png`

### BCM Plasticity

#### `bcm_receptive_fields.py` - Orientation Selectivity through Activity-Dependent Plasticity

This example demonstrates how **BCM (Bienenstock-Cooper-Munro) Rule** enables neurons to develop **orientation-selective
receptive fields** from exposure to oriented bar stimuli, mimicking the development of visual cortex neurons.

**What it does:**

- Generates 1000 oriented bar stimuli at different angles (0° to 180°)
- Trains 4 neurons to develop selective responses using BCM plasticity
- Visualizes learned receptive fields and orientation tuning curves

**Why BCM Rule?**

Pure Hebbian learning has **no selectivity mechanism** - neurons strengthen connections to all inputs they see. BCM adds
a **sliding threshold** that creates two learning regimes:

```
ΔW = η · y · (y - θ) · x^T
            ↑
    BCM modulation function φ(y, θ)

where θ = ⟨y²⟩  (sliding threshold adapts to recent activity)
```

**The key insight:** The threshold θ divides learning into:

- **LTP zone** (y > θ): Potentiation - strengthen active synapses
- **LTD zone** (y < θ): Depression - weaken active synapses
- **θ adapts** based on recent activity history (exponential moving average of y²)

**Why This Creates Selectivity:**

1. **Initial random responses**: All neurons respond weakly to all orientations
2. **One orientation triggers stronger response** (by chance): y > θ → LTP
3. **Other orientations trigger weaker response**: y < θ → LTD
4. **Threshold increases** as neuron becomes more active
5. **Neuron becomes selective** to the orientation that consistently drives y > θ

This is a form of **self-organized competitive learning** where neurons naturally differentiate to respond to different
features.

**Expected Results:**

1. **Threshold Evolution** (top-left plot):
    - Thresholds start at ~0.1 and **adapt dynamically**
    - Neurons with stronger responses develop **higher thresholds**
    - Some neurons may have very low thresholds if they don't find a preferred feature
    - **Typical range after training**: 0.001 to 1.5

2. **Weight Magnitude Evolution** (top-right plot):
    - Weight norms **gradually increase** as neurons strengthen preferred connections
    - Different neurons may converge to different magnitudes
    - **No explosion** because LTD balances LTP through the threshold mechanism

3. **Learned Receptive Fields** (bottom 4 plots):
    - Each neuron develops an **oriented bar-like pattern**
    - Different neurons prefer **different orientations**
    - Receptive fields show clear **positive (red) and negative (blue) regions**
    - Structure emerges from random initialization through BCM learning

4. **Orientation Tuning Curves** (separate plot):
    - Each neuron shows **peaked response** at its preferred orientation
    - Different neurons prefer **different angles** (e.g., 45°, 67.5°, 90°)
    - Tuning curves have characteristic **cosine-like shape**
    - This demonstrates **orientation selectivity** analogous to V1 neurons

**Why This Matters:**

BCM Rule explains how **selectivity emerges** in sensory cortex during development:

- V1 neurons in visual cortex become orientation-selective through visual experience
- The sliding threshold provides **homeostatic plasticity** - neurons regulate their own excitability
- This is a more biologically realistic model than Hebbian learning alone

**Biological Relevance:**

- **Critical period**: BCM predicts a developmental window where visual deprivation affects selectivity
- **Contrast threshold**: The θ mechanism matches physiological observations of adaptation
- **Stable selectivity**: LTP and LTD balance prevents runaway excitation

**Parameters to Experiment:**

- `learning_rate`: Smaller values (0.00001) give more stable learning
- `threshold_tau`: Controls how fast θ adapts (higher = slower adaptation)
- `n_epochs`: More epochs allow stronger differentiation
- `n_neurons`: More neurons can learn different orientations

**What Might Go Wrong:**

- **All neurons learn same orientation**: Learning rate too high or too few training examples
- **No clear selectivity**: Learning rate too low or threshold_tau too small
- **Weight explosion**: Learning rate too high (add weight clipping if needed)

**Trainer**: `BCMTrainer`
**Model**: `LinearLayer` (with `use_bcm_threshold=True`)
**Outputs**: `bcm_receptive_fields.png`, `bcm_orientation_tuning.png`

### STDP (Spike-Timing-Dependent Plasticity)

#### `stdp_temporal_learning.py` - Temporal Pattern Learning through Spike Timing

This example demonstrates how **STDP (Spike-Timing-Dependent Plasticity)** enables spiking neural networks to learn
temporal patterns based on **precise spike timing**, implementing the biological principle that "neurons that spike
together in sequence, wire together."

**What it does:**

- Generates temporal spike patterns with time-varying input activity
- Trains spiking neurons (LIF model) using STDP to learn spike correlations
- Visualizes weight evolution showing timing-dependent changes
- Demonstrates causal vs. anti-causal spike pairing effects

**Why STDP?**

All previous rules (Hebbian, Oja, BCM) are **rate-based** - they depend on average firing rates. STDP is **timing-based
** - it depends on **millisecond-precise spike times**:

```
ΔW_ij = A_plus * trace_pre[j] * spike_post[i] - A_minus * trace_post[i] * spike_pre[j]
         ↑                                         ↑
    LTP (causality)                           LTD (anti-causality)

where:
  trace = decay * trace + spike  (exponential spike history)
```

**The Temporal Credit Assignment:**

STDP solves the **temporal credit assignment problem** - which pre-synaptic spike caused which post-synaptic spike?

```
Timeline:  t=0      t=10     t=20
           ↓        ↓        ↓
Pre-syn:   ⚡       •        •         (spike at t=0)
Post-syn:  •        ⚡       •         (spike at t=10)

Result: LTP (strengthen) - pre caused post
```

vs.

```
Timeline:  t=0      t=10     t=20
           ↓        ↓        ↓
Pre-syn:   •        ⚡       •         (spike at t=10)
Post-syn:  ⚡       •        •         (spike at t=0)

Result: LTD (weaken) - pre didn't cause post
```

**How Spike Traces Implement This:**

Instead of computing time differences Δt = t_post - t_pre, STDP uses **exponential traces**:

1. **Pre-synaptic trace**: Decays after each pre-spike
    - `trace_pre = 0.9 * trace_pre + spike_pre`
    - Creates a "memory" of recent pre-synaptic activity

2. **Post-synaptic trace**: Decays after each post-spike
    - `trace_post = 0.9 * trace_post + spike_post`
    - Creates a "memory" of recent post-synaptic activity

3. **Weight updates**:
    - **LTP**: When post-spike occurs, strengthen based on current pre-trace
        - If pre spiked recently → high trace → strong LTP
    - **LTD**: When pre-spike occurs, weaken based on current post-trace
        - If post spiked recently → high trace → strong LTD

**Expected Results:**

1. **Weight Changes by Input Group**:
    - **Early inputs (0-4)**: Weight increase (LTP dominant)
      → Consistently spike before outputs, creating causal relationship
    - **Middle inputs (5-9)**: Moderate changes
      → Mixed timing relationship with outputs
    - **Late inputs (10-14)**: Weight decrease or minimal change (LTD)
      → Often spike after outputs have already spiked
    - **Noise inputs (15-19)**: Minimal change
      → No consistent temporal correlation

2. **Initial vs Final Weights** (heatmaps):
    - **Initial**: Random, uniform distribution around 0.025
    - **Final**: Structure emerges - early inputs have stronger weights
    - **ΔW**: Clear positive changes for early inputs, negative for late inputs

3. **Spike Rasters**:
    - **Input raster**: Three distinct temporal groups visible (0-10, 10-20, 20-30 timesteps)
    - **Output raster**: Neurons spike primarily in response to early inputs
    - This demonstrates **temporal selectivity** - neurons learn to respond to specific timing patterns

4. **Weight Evolution** (bottom-right plot):
    - **Early input weights**: Gradual increase over training
    - **Middle input weights**: Relatively stable
    - **Late input weights**: May decrease slightly
    - Shows that STDP creates **temporal preference** for causal inputs

**Why This Matters:**

STDP explains:

- **Temporal sequence learning**: How brain circuits learn time-ordered patterns
- **Predictive coding**: Early signals predict later events
- **Delay tuning**: Neurons become selective for specific spike timing relationships
- **Synaptic competition**: Synapses compete based on timing, not just correlation

**Biological Relevance:**

1. **Discovery**: Bi & Poo (1998) measured STDP in hippocampal neurons
    - Pre before post (+10ms) → LTP
    - Post before pre (-10ms) → LTD
    - Window: ~20-40ms in biology

2. **Function in the brain**:
    - **Auditory cortex**: Temporal sequence learning for speech
    - **Motor cortex**: Action sequence learning
    - **Hippocampus**: Temporal order memory (place cell sequences)
    - **Cerebellum**: Precise timing for motor control

3. **Computational advantages**:
    - **Local rule**: Only needs pre/post spike times, no global coordination
    - **Unsupervised**: No error signal required
    - **Predictive**: Naturally learns cause-effect relationships

**Key Parameters:**

- `learning_rate`: Global scaling of weight changes (default: 0.02)
- `A_plus`: LTP magnitude (default: 0.005)
- `A_minus`: LTD magnitude (default: 0.00525, slightly > A_plus)
    - **Why A_minus > A_plus?**: Creates competition and prevents runaway potentiation
- `trace_decay`: Temporal window (default: 0.90)
    - Higher → longer memory of spike history
    - Lower → only recent spikes matter
- `threshold`: Spike threshold (default: 0.8)
    - Lower → more spiking activity
    - Higher → sparse spiking

**What Might Go Wrong:**

- **All weights increase**: A_plus >> A_minus, learning rate too high
    - Solution: Set A_minus ≥ A_plus, reduce learning rate
- **No weight changes**: Threshold too high (no spikes), learning rate too low
    - Solution: Lower threshold to 0.5-0.8, increase learning rate
- **Unstable weights**: Learning rate too high
    - Solution: Reduce to 0.01-0.02
- **No temporal structure**: trace_decay too low
    - Solution: Increase to 0.90-0.95 for longer temporal integration

**Comparison with Rate-Based Rules:**

| Property        | Rate-Based (Hebbian, BCM)     | STDP (Spike-Timing)          |
|-----------------|-------------------------------|------------------------------|
| Time resolution | Coarse (~100ms windows)       | Fine (~1ms precision)        |
| Information     | Average firing rate           | Precise spike times          |
| Causality       | No temporal order             | Pre→post order matters       |
| Biological      | Simplified abstraction        | Closer to biology            |
| Computation     | Faster (smoother signals)     | Slower (event-driven)        |
| Use cases       | Rate coding, PCA, selectivity | Temporal patterns, sequences |

**Trainer**: `STDPTrainer`
**Model**: `SpikingLayer` (LIF neurons with spike traces)
**Output**: `stdp_temporal_learning.png`

## Running Examples

All examples can be run independently:

```bash
cd examples/brain_inspired/

# Hebbian learning examples
python hopfield_train.py
python hopfield_train_1d.py
python hopfield_hebbian_vs_antihebbian.py

# Energy-based learning
python hopfield_energy_diagnostics.py

# Normalized Hebbian (Oja)
python oja_pca_extraction.py

# BCM plasticity
python bcm_receptive_fields.py

# STDP (spike-timing-dependent plasticity)
python stdp_temporal_learning.py
```

### Dependencies

```bash
pip install numpy matplotlib scikit-learn scikit-image
```

## Algorithm Summary

| Algorithm                | Model                | Trainer            | Analyzer         | Key Features                                               |
|--------------------------|----------------------|--------------------|------------------|------------------------------------------------------------|
| **Hebbian**              | AmariHopfieldNetwork | HebbianTrainer     | -                | Pattern storage, associative memory                        |
| **Anti-Hebbian**         | AmariHopfieldNetwork | AntiHebbianTrainer | -                | Decorrelation, competitive learning                        |
| **Hopfield Diagnostics** | AmariHopfieldNetwork | HebbianTrainer     | HopfieldAnalyzer | Energy minimization, capacity analysis, recall diagnostics |
| **Oja**                  | LinearLayer          | OjaTrainer         | -                | PCA, weight normalization                                  |
| **BCM**                  | LinearLayer          | BCMTrainer         | -                | Sliding threshold, receptive field development             |
| **STDP**                 | SpikingLayer         | STDPTrainer        | -                | Spike-timing plasticity, temporal learning                 |

## Key Concepts

### Hebbian Learning

**"Neurons that fire together, wire together"** - Strengthens connections between co-active neurons.

**Rule**: `ΔW = η · y · x^T`

The original and most fundamental learning rule in neuroscience. When a presynaptic neuron (x) and postsynaptic neuron (
y) are simultaneously active, their connection strengthens. This leads to:

- **Associative memory**: Patterns become attractors in network dynamics
- **Pattern completion**: Partial cues can retrieve full patterns
- **Problem**: Weights grow unbounded without additional mechanisms

### Anti-Hebbian Learning

**"Neurons that fire together, wire apart"** - Weakens connections for decorrelation and competitive learning.

**Rule**: `ΔW = -η · y · x^T`

The opposite of Hebbian learning. Used for:

- **Decorrelation**: Remove redundancies in neural representations
- **Competitive learning**: Neurons compete to represent different features
- **Lateral inhibition**: Models inhibitory connections in cortex
- **Sparse coding**: Encourages sparse, efficient representations

### Oja's Rule - Normalized Hebbian Learning

**Rule**: `ΔW = η · (y·x^T - y²·W)`

Oja's Rule solves the **weight explosion problem** of pure Hebbian learning by adding a **weight decay term**
proportional to the output squared. This creates a natural competition between Hebbian growth and normalization.

**Mathematical Beauty:**

The rule can be derived from constrained optimization:

```
Maximize: ⟨(W^T x)²⟩  (capture variance)
Subject to: ||W|| = 1   (unit norm constraint)
```

The weight decay term `-y²·W` emerges as the Lagrange multiplier enforcing the constraint.

**What it achieves:**

1. **Automatic normalization**: Weights converge to unit vectors without explicit normalization
2. **PCA extraction**: Each neuron extracts a principal component
3. **Orthogonality**: Multiple neurons extract different components (with additional mechanisms)
4. **Biological plausibility**: Simpler than computing covariance matrices

**Convergence theorem**: For a single neuron, Oja's Rule converges to the first principal component (eigenvector with
largest eigenvalue) of the input covariance matrix.

**Extensions**:

- **Sanger's Rule**: Multiple neurons extract orthogonal PCs using Gram-Schmidt-like mechanism
- **Generalized Hebbian Algorithm (GHA)**: Extracts multiple PCs with lateral connections

### BCM Rule - Sliding-Threshold Plasticity

**Rule**:

```
ΔW = η · φ(y, θ) · x^T
where φ(y, θ) = y(y - θ)  (BCM modulation function)
      θ = ⟨y²⟩            (sliding threshold)
```

BCM (Bienenstock-Cooper-Munro, 1982) extends Hebbian learning with a **dynamic threshold** that creates
history-dependent plasticity.

**The BCM Modulation Function:**

The function `φ(y, θ) = y(y - θ)` creates a **nonlinear learning rule**:

```
     φ(y)
      ^
  LTP |     /
      |    /
  ----+---/----> y
      |  /  θ
  LTD | /
```

**Key properties**:

1. **Super-linear**: Weight changes grow faster than linearly with y
2. **Sign change at θ**: Switch from depression to potentiation
3. **Stable fixed point**: Neurons self-stabilize at moderate activity levels

**Why the sliding threshold?**

Fixed threshold would fail because:

- Too low → all neurons potentiate → no selectivity
- Too high → all neurons depress → network dies

**Adaptive threshold** `θ = ⟨y²⟩` creates **homeostatic plasticity**:

- High activity → θ increases → harder to trigger LTP → activity decreases
- Low activity → θ decreases → easier to trigger LTP → activity increases

**Selectivity Mechanism:**

BCM creates **competitive learning** without lateral connections:

1. Random initialization → all neurons weakly respond to all inputs
2. One input pattern accidentally triggers `y > θ` → LTP for that pattern
3. Other patterns trigger `y < θ` → LTD for those patterns
4. Neuron becomes **selective** for the first pattern
5. Different neurons become selective for different patterns

**Theoretical Results:**

- **Selectivity theorem**: Under natural image statistics, BCM leads to orientation selectivity
- **Critical period**: BCM predicts developmental plasticity windows observed in visual cortex
- **Ocular dominance**: BCM explains monocular deprivation effects

**Biological Support:**

- LTP/LTD crossover frequency matches BCM predictions
- Developmental timeline matches critical period in V1
- Contrast adaptation matches threshold sliding

**Modern Extensions:**

- **Spike-timing-dependent BCM**: Combines BCM with precise timing
- **Network BCM**: Multiple neurons with lateral inhibition
- **Deep BCM**: Multi-layer BCM for hierarchical features

### STDP Rule - Spike-Timing-Dependent Plasticity

**Rule**:

```
ΔW_ij = A_plus * trace_pre[j] * spike_post[i] - A_minus * trace_post[i] * spike_pre[j]

where:
  trace_pre[j] = decay * trace_pre[j] + spike_pre[j]   (pre-synaptic trace)
  trace_post[i] = decay * trace_post[i] * spike_post[i] (post-synaptic trace)
```

STDP is the **first temporal learning rule** that makes the **order of spikes** matter, not just their co-occurrence.

**The Temporal Asymmetry:**

STDP implements a fundamental biological principle discovered by Bi & Poo (1998):

```
Pre → Post (+Δt):  LTP (strengthen)  "pre caused post"
Post → Pre (-Δt):  LTD (weaken)      "pre didn't cause post"
```

This creates a **causality detector** - synapses strengthen when presynaptic activity **predicts** postsynaptic
activity.

**How It Works:**

Instead of computing precise time differences, STDP uses **eligibility traces**:

1. **Spike traces as memory**: Each spike leaves an exponentially decaying trace
    - Pre-spike sets `trace_pre ← trace_pre + 1`, then decays
    - Post-spike sets `trace_post ← trace_post + 1`, then decays

2. **Trace-based updates**:
    - When **post-spike** occurs: `ΔW ∝ trace_pre` (recent pre activity)
        - High pre-trace → pre spiked recently → causal → LTP
    - When **pre-spike** occurs: `ΔW ∝ -trace_post` (recent post activity)
        - High post-trace → post spiked recently → anti-causal → LTD

3. **Temporal window**: Trace decay constant determines learning window
    - Biology: ~20-40ms window
    - Implementation: `decay = 0.90-0.95` creates similar windows

**Why This Matters:**

STDP enables:

- **Sequence learning**: Neurons learn temporal order of events
- **Predictive wiring**: Connections strengthen from early to later signals
- **Coincidence detection**: Becomes selective for specific timing relationships
- **Delay lines**: Can learn to respond to specific inter-spike intervals

**Biological Evidence:**

**Original Discovery** (Bi & Poo, 1998):

- Hippocampal cultures show precise timing window
- +10ms (pre before post): ~40% LTP
- -10ms (post before pre): ~30% LTD
- Outside ±40ms window: no change

**Brain Regions:**

- **Hippocampus**: Temporal sequence encoding (place cell sequences, theta phase precession)
- **Auditory cortex**: Speech/sound sequence learning
- **Motor cortex**: Action sequence planning
- **Visual cortex**: Direction selectivity (motion sequences)
- **Cerebellum**: Precise timing for motor control

**Theoretical Insights:**

**Information maximization**: STDP maximizes temporal information transfer between neurons by strengthening predictive
connections.

**Stability mechanisms**:

- **A_minus ≥ A_plus**: LTD slightly stronger prevents runaway potentiation
- **Weight bounds**: Hard bounds (0, w_max) prevent unlimited growth
- **Trace normalization**: Decay balances potentiation and depression

**Comparison with BCM**:

| Feature     | BCM                           | STDP                             |
|-------------|-------------------------------|----------------------------------|
| Signal type | Firing rates                  | Precise spike times              |
| Time scale  | ~100ms (rate window)          | ~10ms (spike timing)             |
| Computation | Continuous values             | Discrete events                  |
| Causality   | No temporal order             | Pre→post matters                 |
| Homeostasis | Sliding threshold θ           | Weight bounds + A_minus ≥ A_plus |
| Use case    | Selectivity, receptive fields | Sequences, prediction            |

Both solve the **stability problem** differently:

- **BCM**: Adaptive threshold creates homeostasis
- **STDP**: Anti-causal depression balances causal potentiation

**Extensions:**

- **Triplet STDP**: Includes three-spike interactions for better stability
- **Voltage-dependent STDP**: Incorporates postsynaptic depolarization
- **Dopamine-modulated STDP**: Reward modulates plasticity
- **Symmetric STDP**: Both directions strengthen (for auto-association)

**Computational Advantages:**

1. **Local rule**: Only needs pre/post spike times at each synapse
2. **Event-driven**: Computes only when spikes occur (efficient)
3. **Unsupervised**: No error signal or teacher required
4. **Causal**: Naturally learns cause-effect relationships

**Challenges:**

- **Sparse events**: Fewer updates than rate-based rules
- **Noise sensitivity**: Single spikes can trigger updates
- **Parameter tuning**: Trace decay and A_plus/A_minus ratios critical
- **Slow learning**: May need many examples for stable learning

### Energy-Based Models

Models that minimize energy functions (Hopfield) for pattern completion and associative recall.

**Energy Function**: `E = -∑ᵢⱼ Wᵢⱼ sᵢ sⱼ`

Dynamics evolve to minimize energy, creating attractor dynamics where patterns become stable states. Used for:

- **Content-addressable memory**: Retrieve by partial information
- **Denoising**: Remove corruption by relaxing to nearest attractor
- **Constraint satisfaction**: Find states satisfying multiple constraints

## References

### Classic Papers

**Hebbian Learning:**

- **Hebb, D. O. (1949)**. *The Organization of Behavior*. Wiley, New York.
    - Original formulation: "Cells that fire together wire together"

**Hopfield Networks:**

- **Hopfield, J. J. (1982)**. *Neural networks and physical systems with emergent collective computational abilities*.
  PNAS, 79(8), 2554-2558.
    - Energy-based associative memory using symmetric weights

**Oja's Rule:**

- **Oja, E. (1982)**. *Simplified neuron model as a principal component analyzer*. Journal of Mathematical Biology, 15(
  3), 267-273.
    - Proves convergence to first principal component
    - Shows weight normalization emerges automatically

**BCM Theory:**

- **Bienenstock, E. L., Cooper, L. N., & Munro, P. W. (1982)**. *Theory for the development of neuron selectivity:
  orientation specificity and binocular interaction in visual cortex*. Journal of Neuroscience, 2(1), 32-48.
    - Introduces sliding threshold mechanism
    - Explains orientation selectivity development
    - Predicts critical period effects

**STDP (Spike-Timing-Dependent Plasticity):**

- **Bi, G. Q., & Poo, M. M. (1998)**. *Synaptic modifications in cultured hippocampal neurons: dependence on spike
  timing, synaptic strength, and postsynaptic cell type*. Journal of Neuroscience, 18(24), 10464-10472.
    - First experimental demonstration of STDP
    - Shows asymmetric learning window (±40ms)
    - Establishes pre→post = LTP, post→pre = LTD

- **Gerstner, W., & Kistler, W. M. (2002)**. *Spiking Neuron Models: Single Neurons, Populations, Plasticity*. Cambridge
  University Press.
    - Comprehensive theoretical treatment of spiking neurons
    - Chapter 10 covers STDP in detail

- **Morrison, A., Diesmann, M., & Gerstner, W. (2008)**. *Phenomenological models of synaptic plasticity based on spike
  timing*. Biological Cybernetics, 98(6), 459-478.
    - Reviews different STDP formulations
    - Discusses trace-based implementations

### Additional Reading

**Oja's Rule Extensions:**

- **Sanger, T. D. (1989)**. *Optimal unsupervised learning in a single-layer linear feedforward neural network*. Neural
  Networks, 2(6), 459-473.
    - Generalized Hebbian Algorithm (GHA) for multiple PCs

**BCM Extensions:**

- **Intrator, N., & Cooper, L. N. (1992)**. *Objective function formulation of the BCM theory of visual cortical
  plasticity*. Neural Networks, 5(1), 3-17.
    - Statistical mechanics formulation of BCM

- **Shouval, H. Z., Bear, M. F., & Cooper, L. N. (2002)**. *A unified model of NMDA receptor-dependent bidirectional
  synaptic plasticity*. PNAS, 99(16), 10831-10836.
    - Connects BCM to NMDA receptor dynamics

**Biological Evidence:**

- **Hubel, D. H., & Wiesel, T. N. (1962)**. *Receptive fields, binocular interaction and functional architecture in the
  cat's visual cortex*. Journal of Physiology, 160(1), 106-154.
    - Discovery of orientation selectivity in V1

- **Blais, B. S., Intrator, N., Shouval, H., & Cooper, L. N. (1998)**. *Receptive field formation in natural scene
  environments: comparison of single-cell learning rules*. Neural Computation, 10(7), 1797-1813.
    - BCM outperforms other rules for natural images

### Online Resources

- **Scholarpedia: BCM Theory**: http://www.scholarpedia.org/article/BCM_theory
- **Scholarpedia: Oja Learning Rule**: http://www.scholarpedia.org/article/Oja_learning_rule
- **Neural Networks Course (Stanford CS231n)**: Covers biological inspiration
- **Theoretical Neuroscience (Dayan & Abbott)**: Chapter 8 covers plasticity rules

## Tips & Troubleshooting

### Getting Started

1. **Start simple**: Try `oja_pca_extraction.py` first - it runs quickly and shows clear results
2. **Then BCM**: `bcm_receptive_fields.py` takes longer but shows beautiful emergent selectivity
3. **Compare**: Run `hopfield_hebbian_vs_antihebbian.py` to see the difference between learning rules
4. **Visualization**: All examples save plots automatically - no need for display

### Common Issues & Solutions

**Oja's Rule Problems:**

| Problem                 | Possible Cause            | Solution                          |
|-------------------------|---------------------------|-----------------------------------|
| Weights don't normalize | `normalize_weights=False` | Set to `True` in OjaTrainer       |
| Poor PCA alignment      | Too few epochs            | Increase `n_epochs` to 50+        |
| Slow convergence        | Learning rate too small   | Increase `learning_rate` to 0.01  |
| Oscillating weights     | Learning rate too large   | Decrease `learning_rate` to 0.001 |

**BCM Problems:**

| Problem                     | Possible Cause            | Solution                           |
|-----------------------------|---------------------------|------------------------------------|
| No selectivity emerges      | Learning rate too low     | Increase to 0.0001                 |
| All neurons same            | Not enough diversity      | Increase `n_epochs` or `n_samples` |
| Weight explosion            | Learning rate too high    | Decrease to 0.00001                |
| Threshold stuck at 0.1      | No activity               | Check that inputs are normalized   |
| Neurons don't differentiate | `threshold_tau` too small | Increase to 100+                   |

**General Tips:**

- **Learning rate**: Start small (0.0001 for BCM, 0.01 for Oja) and adjust
- **Epochs**: More epochs → stronger learning, but diminishing returns after ~50
- **Batch size**: Process all data per epoch for stable learning
- **Initialization**: Random seeds affect BCM selectivity - try different seeds
- **Monitoring**: Watch threshold evolution (BCM) or weight norms (Oja) for signs of problems

### Understanding the Outputs

**Oja's PCA Extraction:**

- **Good sign**: Weight norms stay at 1.0, cosine similarity > 0.9 with sklearn PCA
- **Bad sign**: Weight norms diverge, poor alignment with PCA
- **Debug**: Check that `normalize_weights=True` and learning rate is appropriate

**BCM Receptive Fields:**

- **Good sign**: Each neuron has distinct preferred orientation, clear tuning curves
- **Bad sign**: All neurons respond equally to all orientations
- **Debug**: Try different random seeds, increase epochs, adjust `threshold_tau`

**Orientation Tuning:**

- **Expected**: Peaked curves centered at different angles (0-180°)
- **Width**: ~30-60° at half-maximum (similar to biological V1)
- **Diversity**: Different neurons prefer different angles

### Parameters to Experiment With

**For deeper understanding, try varying:**

1. **Data structure** (Oja):
    - Change variance ratios of components
    - Add more or fewer principal components
    - Increase noise level

2. **Network size** (BCM):
    - More neurons (8-16) → more orientation coverage
    - Fewer neurons (2) → clearer competition

3. **Learning dynamics**:
    - Compare fast vs. slow learning (10x difference)
    - Try online learning (1 sample at a time)
    - Implement learning rate decay

4. **Biological realism**:
    - Add weight bounds (clip to [-1, 1])
    - Implement sparse connectivity
    - Add noise to updates

### Advanced Explorations

**Oja's Rule:**

- Implement **Sanger's Rule** for orthogonal PC extraction
- Compare with **standard PCA** on real images
- Test on **high-dimensional datasets** (MNIST pixels)
- Measure **convergence speed** vs. learning rate

**BCM Rule:**

- Test on **natural image patches** instead of bars
- Implement **lateral inhibition** for stronger competition
- Measure **critical period** by training at different ages
- Try **non-stationary inputs** (changing statistics)

### Performance Expectations

**Timing:**

- `oja_pca_extraction.py`: ~10-20 seconds (20 epochs, 500 samples)
- `bcm_receptive_fields.py`: ~2-3 minutes (100 epochs, 1000 samples)
- Hopfield examples: ~5-30 seconds depending on network size

**Memory:**

- Oja: Minimal (~10 MB for 50-dim data)
- BCM: Moderate (~50 MB for 144-dim inputs)
- Scales linearly with input dimensionality

**Typical Results:**

- Oja: Cosine similarity 0.95-0.99 with PCA
- BCM: 3-4 distinct orientations from 4 neurons
- Both: Stable and reproducible with fixed random seed
