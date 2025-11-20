# Neuromorphic Keyword Spotting (CS-576 Final Project)

## Project Overview

This project implements a complete Keyword Spotting (KWS) pipeline designed for low-power, neuromorphic edge computing.  
The objective is to train a conventional deep learning model for speech command recognition, convert it to a Spiking Neural Network (SNN), and evaluate it using an Intel Loihi–compatible neuromorphic emulator.

The end-to-end workflow includes:

1. Baseline CNN training on MFCC features  
2. SNN conversion using snnTorch  
3. Loihi-based neuromorphic emulation using Nengo and Nengo-Loihi  
4. Accuracy and energy-style comparisons between ANN, SNN, and Loihi models

This repository provides a reproducible implementation of all stages.

---

## Phase 1: Baseline CNN (Completed)

The baseline model is a conventional Convolutional Neural Network trained on a subset of the Google Speech Commands v0.02 dataset. It reaches strog accuracy and serves as the foundation for SNN conversion and Loihi emulation.

**Dataset classes**  
`yes`, `no`, `go`, `stop`, `up`, `down`

**Feature extraction**  
Mel-Frequency Cepstral Coefficients (MFCC), normalized per sample.

**Model architecture**
- Conv2D + ReLU + MaxPool  
- Conv2D + ReLU + MaxPool  
- Fully connected layers (64 → 6)

**Training details**
- Optimizer: Adam  
- Scheduler: StepLR  
- Epochs: 10

**Final CNN Performance (Full Test Set)**
| Metric | Result |
|--------|--------|
| Training Accuracy | ~98% |
| Validation Accuracy | ~85% |
| Test Accuracy (Full Dataset) | ~80-85% |
| Test Accuracy (Loihi Evaluation Subset, 50 samples) | ~44% |

**Saved model**  
`saved_models/baseline_cnn_kws_vfinal.pt`

**Notes**
1. The drop from 85% -> 44% occurs because the Loihi evaluation was performed on a random 50-sample subset, not the whole test set
2. The CNN still significantly outperforms the SNN and Loihi classifier

---

## Phase 2: Baseline SNN (Completed)

The SNN is trained from sratch using snnTorch.

**Final SNN Training Results**
| Metric | Result |
|--------|--------|
| Test Loss | 1.793 |
| Training Accuracy | ~15-16% |
| Spike Count (per batch) | ~643,272 total spikes |
| Layer 1 Spikes | 523,184 |
| Layer 2 Spikes | 119,926 |
| Layer 3 Spikes | 162 |

**Notes**
1. The SNN fires very sparsely in deeper layers, which reflects expected SNN behavior
2. The accuracy is lower than the CNN beuaes training SNNs is harder and MFCCs are rate-based features optimized for ANNs 

---

## Phase 3: SNN Conversion (Completed)

The trained CNN is converted into a spiking model using snnTorch.

Key modifications:
- ReLU replaced by Leaky-Integrate-and-Fire (LIF) neurons  
- Surrogate gradient (fast sigmoid) for backprop-compatible spiking  
- Rate coding used for temporal spike generation  
- Weights copied directly from the baseline CNN  
- Multi-timestep inference (T = 10–100)

**SNN parameters**
- β ∈ {0.90, 0.95, 0.97, 0.99}  
- Timesteps ∈ {10, 25, 50, 75, 100}

**Saved model**  
`saved_models/snn_kws_beta0.95_T50.pt`

---

## Phase 4: Loihi-Compatible CNN Emulation (Completed)

Using Nengo and Nengo-Loihi, the final SNN head is executed on a Loihi-style emulator:

- Converts CNN features into a 64-dimensional rate-coded input  
- Runs these features through a LIF-based classifier on the Loihi backend  
- Transfers classification weights directly from the CNN  
- Evaluates temporal spike outputs to obtain final logits  
- Supports full dataset evaluation via DataLoader

**Outcome**
- Functional Loihi emulation of the classification head  
- Demonstrated spike-driven inference using a neuromorphic backend  
- Accuracy observed is lower than CNN baseline, but consistent with event-driven spiking behavior

---

## Phase 5: Loihi-Compatible SNN Emulation (Completed)

Using Nengo and Nengo-Loihi, the final SNN's 64-unit FC head is mapped to a LIF ensemble and evaluated on Loihi-like neuromorphic hardware emulator.

**Final Loihi SNN Classifier Performance (50-sample Eval)**

| Model | Accuracy |
|--------|-----------|
| SNN (PyTorch Forward Pass) | 24% |
| SNN Loihi Emulated Classifier | 14% |

**Notes**
1. Loihi classifier accuracy is lower because:
   a. The Loihi model uses rate-coded static inputs (not temporal spikes)
   b. Mapping weights to Loihi LIF neurons introduce synapse constraints and quantization
   c. Only the final classifier head is emulated, not the full SNN
---


## Phase 6: Final Comaprison ANN vs SNN vs Loihi Evaluation (Completed)

A dedicated comparative pipeline evaluates:

1. ANN (CNN) accuracy  
2. SNN (converted model) spike-driven accuracy  
3. Loihi emulated accuracy  
4. Energy-style proxy metrics based on spike counts

**Example preliminary results**  
(50 random samples, demonstration only)

| Model | Accuracy (Full Dataset) | Accuracy (Loihi Evaluation Subset) | Notes | 
|--------|-----------|-----------|-----------|
| CNN (Baseline) | ~80-85% | 44% | Strongest Baseline |
| SNN (PyTorch) | 16% | 24% (Subset) | Much lower accuracy but highly sparse |
| Loihi CNN -> SNN Classifier | - | 20% | Uses CNN features & Loihi LIF ensemble |
| Loihi SNN Classifier | - | 14% | Hardest challenge: spike -> rate mapping |

More detailed experiments may improve this result through:
- Parameter tuning  
- Additional normalization steps  
- Alternative coding schemes  

---

## Energy Proxy (Spike-Based Efficiency)
Spiking neural networks trade accuracy for potential energy efficiency.

SNN Spike Summary:
1. Total mspikes in 1 batch: ~643k
2. But spikes are:
   a. large in ealy convolution layers (input-driven)
   b. extremely sprase in deeper layers (162 spikes in layer 3)
3. Proves that neuromorphic computing is good at:
   a. early feature extraction -> dense
   b. later layers -> sparse computation
     
## Environment and Requirements

Major dependencies:

| Library | Version |
|---------|---------|
| PyTorch | 2.8.0+ |
| snnTorch | 0.9.1 |
| torchaudio | 2.8.0+ |
| Nengo | 4.x |
| Nengo-Loihi | 1.x |
| NumPy | 1.26+ |
| tqdm | Latest |

The project runs on:
- Google Colab (T4 GPU)  
- macOS (M1/M2/M3/M5)  
- Local virtual environments (Conda recommended)

---

## Completed Checklist

- Baseline CNN trained successfully  
- CNN saved and reproducible  
- SNN conversion implemented using snnTorch  
- Temporal rate coding for spike generation  
- Loihi-compatible emulator implemented using Nengo  
- Evaluation pipeline for CNN vs Loihi created  
- Repository cleaned to exclude large datasets  
- Comparative notebook completed  

---

## Future Work

Potential extensions:

- Fine-tuning SNN with backprop-through-time  
- Exploring temporal coding strategies (TTFS, phase coding)  
- Deployment on real Intel Loihi hardware if available  
- Expanding dataset to include more spoken commands  
- Investigating latency-accuracy trade-offs  
- Adding noise robustness evaluation  

---

## Summary

This project demonstrates a complete end-to-end neuromorphic keyword spotting pipline:
1. A high-accuracy Convolutional Neural Network reaches ~80–85% accuracy on SpeechCommands and serves as the ANN baseline
2. A spiking neural network (snnTorch) is trained from scratch using surrogate gradients and achieves ~16% accuracy, with sparse spike activity aligned with neuromorphic computation principles
3. Both the CNN and SNN classification heads are executed on an Intel Loihi–compatible emulator using Nengo and Nengo-Loihi
   a. CNN → Loihi classifier accuracy: 20%
   b. SNN → Loihi classifier accuracy: 14%
4. An evaluation framework compares accuracy, spikes, and performance across ANN, SNN, and neuromorphic Loihi models
   
This illustrates the core trade-off in neuromorphic computing:
lower accuracy but significantly sparser computation, enabling theoretical energy gains. 

The system leverages conventional training, spiking conversion, and Loihi emulation to explore accuracy, spike activity, and energy-style behavior.  
The modular pipeline enables further extensions including improved coding strategies, backprop-trained SNNs, and real-hardware deployment on Intel Loihi 1/2.
