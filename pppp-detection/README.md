# PPPP Detection (Polychronous Pattern Detection)

A computational neuroscience toolkit for detecting and analyzing polychronous patterns in neural spike data. This project implements algorithms to identify temporal sequences of spikes that form repeating patterns in spiking neural networks.

## Overview

This project provides tools for:
- **Pattern Detection**: Identifying polychronous polysynaptic pathway patterns (PPPP) in neural spike trains
- **Synthetic Data Generation**: Creating controlled datasets with embedded temporal patterns for algorithm validation
- **Performance Evaluation**: Measuring detection accuracy against ground truth patterns

## Files

### `scan.py` - Pattern Detection Algorithm

The core detection engine that implements a sophisticated clustering-based approach to identify recurring temporal spike patterns.

**Key Functions:**
- `scan_raster(T_labels, N_labels, window_dim=100)` - Main detection function
- `_get_sim_mats()` - Computes similarity matrices between spike windows
- `_cluster_windows()` - Hierarchical clustering of similar spike patterns
- `_check_seq()` - Validates sequential patterns in clusters

**Algorithm Pipeline:**
1. **Windowing**: Creates temporal windows around each spike event
2. **Similarity Computation**: Calculates pattern similarity using set intersection
3. **Optimization**: Searches across clustering cutoff values to maximize pattern repetitions
4. **Template Extraction**: Builds canonical templates for detected patterns
5. **Timing Recovery**: Identifies precise occurrence times for each pattern

**Parameters:**
- `T_labels`: Array of spike times
- `N_labels`: Corresponding neuron IDs
- `window_dim`: Temporal window size (default: 100 time steps)

**Returns:**
- `pattern_template`: Canonical spike patterns (time, neuron) coordinates
- `all_times`: Occurrence times for each detected pattern
- Performance timing metrics

### `simulate_data.py` - Synthetic Data Generation & Validation

Generates realistic synthetic neural data with embedded polychronous patterns for testing and validation.

**Key Functions:**
- `generate_synthetic_data(params, plot=False)` - Creates synthetic spike data with known patterns
- `plot_raster()` - Visualizes spike rasters and detection results
- `get_acc()` - Computes detection accuracy using cross-correlation
- `check_ground_truth()` - Validates detected patterns against known ground truth

**Synthetic Data Parameters:**
```python
params = {
    'N': 30,           # Number of neurons
    'M': 4,            # Number of spiking motifs
    'D': 20,           # Duration of each motif
    'T': 1000,         # Total simulation time
    'seed': 42,        # Random seed
    'SM_repetitions': 10,     # Repetitions per motif
    'spikes_in_SM': 10,       # Spikes per motif
    'noise': 100       # Background noise spikes
}
```

**Visualization Features:**
- Individual spiking motif patterns
- Motif occurrence timing
- Complete raster plot with overlaid patterns
- Cross-correlation analysis results
- Multi-panel diagnostic plots

## Algorithm Details

### Pattern Detection Method

The algorithm uses a multi-stage approach:

1. **Temporal Windowing**: Each spike creates a temporal window containing nearby spikes
2. **Set-Based Similarity**: Windows are compared using set intersection normalized by minimum set size
3. **Hierarchical Clustering**: Similar windows are grouped using complete linkage clustering
4. **Sequential Validation**: Clusters are validated by checking for consistent temporal sequences
5. **Optimization**: Clustering threshold is optimized to maximize pattern repetitions
6. **Template Extraction**: Final patterns are refined by identifying most common spike combinations

### Performance Metrics

- **Cross-correlation accuracy**: Measures template matching quality
- **Temporal precision**: Evaluates timing accuracy of detected occurrences
- **Detection sensitivity**: Fraction of true patterns successfully identified

## Usage Example

```python
# Generate synthetic data
params = {
    'N': 30, 'M': 4, 'D': 20, 'T': 1000, 'seed': 42,
    'SM_repetitions': 10, 'spikes_in_SM': 10, 'noise': 100
}

# Create synthetic dataset with known patterns
A_dense, A_sparse, _, _, K_dense, _ = generate_synthetic_data(params, plot=True)

# Extract spike times and neuron labels
T_labels = A_sparse[1]  # Spike times
N_labels = A_sparse[0]  # Neuron IDs

# Detect patterns
patterns, times, _, _, _, cutoff = scan_raster(T_labels, N_labels, window_dim=50)

# Validate against ground truth
accuracy, cross_corr, pattern_imgs = check_ground_truth(patterns, K_dense)
print(f"Detection accuracy: {accuracy}")
```

## Dependencies

```python
numpy
scipy
matplotlib
pandas
tqdm
json
```

## Applications

- **Neuroscience Research**: Analyzing temporal patterns in recorded neural activity
- **Memory Studies**: Understanding how neural circuits store temporal sequences
- **Neuromorphic Engineering**: Validating spike-based computing architectures
- **Brain-Computer Interfaces**: Detecting command patterns in neural signals

## Algorithm Performance

The detection algorithm is optimized for:
- **Robustness**: Works with noisy, realistic neural data
- **Scalability**: Handles large-scale spike datasets
- **Precision**: Accurately identifies pattern timing and structure
- **Flexibility**: Adapts clustering parameters automatically

## Research Context

This toolkit supports research in:
- Polychronous neural computation theory
- Spike-timing dependent plasticity (STDP)
- Temporal memory mechanisms in biological networks
- Neuromorphic computing applications

---

**Note**: This implementation is part of ongoing computational neuroscience research focusing on temporal pattern detection in spiking neural networks.
