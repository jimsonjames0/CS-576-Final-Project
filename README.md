
# Neuromorphic Keyword Spotting (CS-576 Final Project)

## 📘 Project Overview
This project implements a **Keyword Spotting (KWS)** system using deep learning for **low-power, neuromorphic edge computing**.  
The goal is to build an energy-efficient model that can detect spoken commands (like “yes”, “no”, “up”, “down”, etc.) and later convert it into a **Spiking Neural Network (SNN)** for deployment on neuromorphic hardware.

The project is divided into two main phases:
1. **Baseline CNN (ANN)** – Train a conventional convolutional neural network for keyword recognition.
2. **SNN Conversion** – Convert the trained ANN into an event-driven spiking model to simulate neuromorphic behavior.

---

## 🧠 Phase 1: Baseline CNN (Completed ✅)
- Implemented end-to-end CNN training on the **Speech Commands Dataset v0.02**
- Preprocessed data with **MFCC features**, normalized for better stability
- Used **Adam optimizer** with a **StepLR scheduler**
- Trained for 10 epochs, achieving:
  - **Training Accuracy:** ~98%
  - **Validation Accuracy:** ~85%
  - **Test Accuracy:** ~85%
- Model saved as `baseline_cnn_kws_vfinal.pt`

---
## 🧾 Current Environment
| Library | Version |
|----------|----------|
| PyTorch | 2.8.0+cu126 |
| Torchaudio | 2.8.0+cu126 |
| NumPy | 1.26+ |
| tqdm | Latest |
| Platform | Google Colab (T4 GPU) / macOS M1 (local) |

---

## ✅ Completed
- [x] Setup of PyTorch + Torchaudio environment  
- [x] Implemented MFCC preprocessing with normalization  
- [x] Built CNN with Conv2D + ReLU + MaxPool + Linear layers  
- [x] Achieved stable 85% accuracy  
- [x] Model saved and version-controlled via GitHub  

---

## 🚀 Next Steps (To-Do)

### 🧩 Phase 2 — SNN Conversion
- [ ] Implement conversion of the CNN to an **SNN** using one of the following:
  - [ ] **snnTorch** (recommended; simple and PyTorch-compatible)
  - [ ] **Norse** (for biologically inspired models)
  - [ ] **Nengo** (for neuromorphic simulation)
- [ ] Simulate neuron firing behavior (LIF/IF neurons)
- [ ] Compare SNN accuracy vs CNN
- [ ] Measure **energy efficiency** or **spike sparsity**

### 📊 Phase 3 — Experimentation and Evaluation
- [ ] Run inference tests on SNN for latency/energy comparison
- [ ] Create visualization of spike raster plots
- [ ] Document trade-offs in accuracy vs energy

### 🧾 Phase 4 — Report & Presentation
- [ ] Create final paper/report (3–5 pages)
- [ ] Prepare slides + demonstration video (optional)

---
=======
# CS-576-Final-Project


