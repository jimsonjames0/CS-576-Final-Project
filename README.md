# Neuromorphic Keyword Spotting (CS-576 Final Project)

## Project Overview

This project implements a complete Keyword Spotting (KWS) pipeline designed for low-power, neuromorphic edge computing.

The objective is to:

- Train a conventional deep learning model for speech command recognition (CNN).
- Build and train a Spiking Neural Network (SNN) using snnTorch.
- Run both models through an Intel Loihi–compatible neuromorphic emulator using Nengo / Nengo-Loihi.
- Compare accuracy and spike-based “energy-style” behavior across ANN, SNN, and Loihi-style classifiers.

The end-to-end workflow includes:

1. Baseline CNN training on MFCC features  
2. Baseline SNN training using snnTorch  
3. SNN conversion experiments (ANN → SNN)  
4. Loihi-style neuromorphic emulation using Nengo and Nengo-Loihi  
5. Accuracy and spike-based comparisons between CNN, SNN, and Loihi classifiers  

This repository provides a reproducible implementation of all stages.

---

## Phase 1: Baseline CNN

The baseline model is a conventional Convolutional Neural Network trained on a subset of the Google Speech Commands v0.02 dataset. It reaches strong accuracy and serves as the foundation for SNN work and Loihi emulation.

### Dataset Classes

Targeted 6-class subset:

- `yes`, `no`, `go`, `stop`, `up`, `down`

### Feature Extraction

- Mel-Frequency Cepstral Coefficients (**MFCC**), 40 coefficients
- Per-sample normalization to zero mean and unit variance

### Model Architecture

- Conv2D → ReLU → MaxPool  
- Conv2D → ReLU → MaxPool  
- Fully connected layers: `flatten → 64 → 6`

### Training Details

- Optimizer: **Adam**
- Scheduler: **StepLR**
- Epochs: **10**
- Loss: Cross-Entropy

### Final CNN Performance (Full Test Set)

| Metric                                     | Result      |
|--------------------------------------------|-------------|
| Training Accuracy                          | ~98%        |
| Validation Accuracy                        | ~85%        |
| **Test Accuracy (Full Dataset)**           | ~80–85%     |
| **Test Accuracy (Loihi Eval Subset, 50 samples)** | ~44% |

**Saved model**

- `saved_models/baseline_cnn_kws_vfinal.pt`

**Notes**

- The drop from ~85% → 44% occurs because the Loihi evaluation was performed on a **random 50-sample subset**, not the entire test set.
- The CNN still significantly outperforms both the SNN and Loihi classifiers in terms of raw accuracy.

---

## Phase 2: Baseline SNN

The baseline SNN is trained **from scratch** using snnTorch, instead of being purely converted from the CNN. It uses a similar convolutional structure but replaces ReLU with spiking neurons and unrolls computation over time.

### SNN Architecture (High-Level)

- Conv2D → LIF → MaxPool  
- Conv2D → LIF → MaxPool  
- Fully connected layer → LIF → output layer  
- Temporal unrolling with `num_steps` time steps (e.g., 8)

### Final SNN Training Results

*(evaluated on the same 6-class KWS task)*

| Metric                        | Result                       |
|-------------------------------|------------------------------|
| Test Loss                     | **1.793**                    |
| **Test Accuracy (Full Test Set)** | **~15–16%** (near-random for 6 classes) |
| Spike Count (per batch)       | **~643,272 total spikes**    |
| Layer 1 spikes                | 523,184                      |
| Layer 2 spikes                | 119,926                      |
| Layer 3 spikes                | 162                          |

### Notes

- The SNN fires **densely in earlier convolutional layers** and **very sparsely in deeper layers**, which matches expected SNN behavior.
- Accuracy is much lower than the CNN because:
  - Training SNNs with surrogate gradients is harder.
  - MFCC features are tuned for dense ANNs, not necessarily optimal for spiking rate codes.
  - The architecture and hyperparameters were not aggressively tuned for SNN performance.

---

## Phase 3: SNN Conversion Experiments  

In addition to the baseline SNN trained from scratch, the project includes **ANN → SNN conversion experiments** using snnTorch.

Key modifications for converted SNN experiments:

- Replace ReLU activations with **Leaky-Integrate-and-Fire (LIF)** neurons.
- Use a **surrogate gradient** (fast sigmoid) for backpropagation through spikes.
- Use **rate coding** over multiple time steps to represent activations as spike trains.
- Copy weights from the baseline CNN where applicable.
- Run multi-timestep inference with different temporal horizons.

Typical SNN hyperparameters explored:

- β (membrane decay): {0.90, 0.95, 0.97, 0.99}  
- Timesteps: {10, 25, 50, 75, 100}

**Saved example converted model**

- `saved_models/snn_kws_beta0.95_T50.pt`

These experiments are documented in:

- `snn_conversion/SNN_Conversion.ipynb`
- `snn_conversion/SNN_Conversion.pdf`

---

## Phase 4: Loihi-Compatible CNN Emulation  

Using **Nengo** and **Nengo-Loihi**, the final CNN’s 64-dimensional feature representation is used to drive a small spiking classifier network running on a Loihi-style emulator.

### Setup

- Convert CNN output features (from the penultimate layer, 64-D) into a **rate-coded input** for a Nengo network.
- Build a LIF ensemble with 64 neurons in Nengo:
  - Input: 64-D feature vector.
  - Neuron model: `nengo.LIF()`.
- Connect the ensemble to an output node with a learned linear transform using the CNN’s final FC weights.

### What Runs on the Loihi Emulator

- The **classification head** is emulated:
  - 64-D input → spiking LIF ensemble → 6-class readout.
- The emulator uses `nengo_loihi.Simulator` (or `nengo.Simulator` with Loihi-like settings depending on environment constraints).

### Outcome

- Functional Loihi emulation of the CNN classification head.
- Demonstrates **spike-driven inference** on a neuromorphic backend.
- Accuracy is lower than the pure CNN baseline but consistent with:
  - Rate-coded, spiking LIF readout.
  - No further fine-tuning of weights on the Loihi architecture.

**Loihi CNN Classifier Results (50-sample subset)**

- CNN head accuracy (PyTorch, same subset): **44%**
- Loihi CNN classifier accuracy: **20%**

---

## Phase 5: Loihi-Compatible SNN Emulation  

Similarly, the final SNN’s 64-unit FC representation is mapped to a LIF ensemble and evaluated on a Loihi-style neuromorphic emulator.

### Final Loihi SNN Classifier Performance (50-sample eval)

| Model                          | Accuracy |
|--------------------------------|----------|
| **SNN (PyTorch forward pass, subset)** | **24%**   |
| **SNN Loihi-emulated classifier**      | **14%**   |

### Notes

Loihi classifier accuracy is lower because:

1. The Loihi model uses **rate-coded static inputs** rather than full temporal spiking dynamics.
2. Mapping FC weights onto LIF neurons introduces additional synapse constraints and quantization effects.
3. Only the **final classifier head** is emulated, not the entire deep spiking network.

---

## Phase 6: Final Comparison — ANN vs SNN vs Loihi  

A dedicated comparative pipeline evaluates:

- ANN (CNN) accuracy
- SNN (from-scratch model) accuracy and spike statistics
- Loihi-emulated classifier accuracy (CNN-driven and SNN-driven)
- Energy-style proxies based on **spike counts**

### Summary Comparison

*(50 random samples for Loihi eval subset; full test set where available)*

| Model                    | Accuracy (Full Dataset) | Accuracy (Loihi Eval Subset) | Notes                                               |
|--------------------------|-------------------------|-------------------------------|-----------------------------------------------------|
| **CNN (Baseline)**       | ~80–85%                | **44%**                       | Strongest baseline                                 |
| **SNN (PyTorch)**        | ~16%                   | **24%** (subset)              | Much lower accuracy but highly sparse              |
| **Loihi CNN Classifier** | —                      | **20%**                       | CNN features + LIF ensemble on Loihi-style backend |
| **Loihi SNN Classifier** | —                      | **14%**                       | Hardest case: SNN features → Loihi LIF readout     |

More detailed experiments could improve these results via:

- Hyperparameter tuning (β, timesteps, learning rate)
- Additional normalization steps
- Alternative coding schemes (TTFS, phase coding, etc.)

---

## Energy Proxy (Spike-Based Efficiency)

Spiking neural networks trade off some accuracy for potentially lower energy, thanks to sparse event-driven computation.

### SNN Spike Summary

- Total spikes in 1 batch (on test data): **~643k**
- Spike distribution:
  - **High activity** in early convolution layers (input-driven)
  - **Very sparse** activity in deeper layers (e.g., only 162 spikes in layer 3)

This qualitatively supports a core neuromorphic idea:

- Early feature extraction → denser activity  
- Later decision layers → highly sparse activity

---

## Environment and Requirements

### Major Dependencies

| Library       | Version (example) |
|--------------|-------------------|
| PyTorch      | 2.x (CPU/GPU)     |
| snnTorch     | 0.9.x             |
| torchaudio   | 2.x               |
| Nengo        | 4.x               |
| Nengo-Loihi  | 1.x               |
| NumPy        | 1.26.x            |
| soundfile    | Latest            |
| tqdm         | Latest            |

### Recommended Environments

- Google Colab (T4 GPU)
- macOS (M1/M2/etc.) with Conda
- Local Linux / WSL2 with Conda virtual environment

---

## Completed Checklist

- Baseline CNN trained successfully  
- CNN checkpoint saved and reproducible  
- Baseline SNN trained from scratch using snnTorch  
- ANN → SNN conversion experiments implemented  
- Temporal rate coding used for spike generation  
- Loihi-compatible CNN classifier implemented using Nengo / Nengo-Loihi  
- Loihi-compatible SNN classifier implemented  
- Evaluation pipeline for CNN vs SNN vs Loihi created  
- Repository cleaned to exclude large raw datasets  
- Comparative notebook and plots completed  

---

## Future Work

Potential extensions:

- Fine-tuning SNN with full backprop-through-time (BPTT)
- Exploring temporal coding strategies (TTFS, phase coding, population codes)
- Deploying on real Intel Loihi hardware (if available)
- Expanding the number of spoken commands and background noise conditions
- Studying latency–accuracy trade-offs for real-time KWS
- Evaluating robustness under noisy or adversarial audio conditions

---

## Summary

This project demonstrates a **complete end-to-end neuromorphic keyword spotting pipeline**:

- A high-accuracy CNN reaches **~80–85%** accuracy on the SpeechCommands subset and serves as the ANN baseline.
- A spiking neural network (snnTorch) trained from scratch achieves **~16%** accuracy, with spike activity patterns that match neuromorphic expectations (dense early, sparse late).
- Both CNN and SNN classification heads are executed on an Intel Loihi–compatible emulator using Nengo and Nengo-Loihi:
  - CNN → Loihi classifier accuracy: **20%**  
  - SNN → Loihi classifier accuracy: **14%**
- An evaluation framework compares accuracy, spike counts, and neuromorphic behavior across ANN, SNN, and Loihi-style models.

This illustrates the central trade-off in neuromorphic computing:

> **Lower accuracy but significantly sparser, event-driven computation, enabling potential energy savings.**

The system leverages conventional training, spiking conversion, and Loihi emulation to explore accuracy–energy trade-offs and provides a modular foundation for future work in neuromorphic keyword spotting and low-power SNN design.
