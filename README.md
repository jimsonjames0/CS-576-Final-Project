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

## Phase 1: Baseline CNN (Training Completed)

The baseline model is a conventional Convolutional Neural Network trained on a subset of the Google Speech Commands v0.02 dataset.

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

**Performance**
| Metric | Result |
|--------|--------|
| Training Accuracy | ~98% |
| Validation Accuracy | ~85% |
| Test Accuracy | ~85% |

**Saved model**  
`saved_models/baseline_cnn_kws_vfinal.pt`

---

## Phase 2: SNN Conversion (Completed)

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

## Phase 3: Loihi-Compatible Neuromorphic Emulation (Completed)

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

## Phase 4: ANN vs SNN vs Loihi Evaluation (Completed)

A dedicated comparative pipeline evaluates:

1. ANN (CNN) accuracy  
2. SNN (converted model) spike-driven accuracy  
3. Loihi emulated accuracy  
4. Energy-style proxy metrics based on spike counts

**Example preliminary results**  
(50 random samples, demonstration only)

| Model | Accuracy |
|--------|-----------|
| CNN | 44% |
| Loihi Head | 20% |

More detailed experiments may improve this result through:
- Parameter tuning  
- Additional normalization steps  
- Alternative coding schemes  

---

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

This project demonstrates a full pipeline for transforming a traditional CNN-based keyword spotting model into a spiking neural network suitable for neuromorphic hardware.  
The system leverages conventional training, spiking conversion, and Loihi emulation to explore accuracy, spike activity, and energy-style behavior.  
The framework is modular and extendable, enabling further research in neuromorphic keyword spotting and low-power SNN design.
