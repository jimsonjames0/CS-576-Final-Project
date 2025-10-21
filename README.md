# 🔊 Neuromorphic Keyword Spotting (CS-576 Final Project)

## 📘 Project Overview
This project implements a **Keyword Spotting (KWS)** system for **low-power, neuromorphic edge computing**.  
The goal is to develop an energy-efficient deep learning pipeline that recognizes spoken commands such as “yes”, “no”, “up”, “down”, etc., and converts the trained model into a **Spiking Neural Network (SNN)** to simulate **neuromorphic hardware behavior**.

The workflow is divided into two core phases:

1. 🧩 **Baseline CNN (ANN)** — Conventional deep network trained for speech recognition.  
2. ⚡ **SNN Conversion** — Converted to a spiking model using rate-coded neurons for event-driven inference and energy evaluation.

---

## 🧠 Phase 1: Baseline CNN (Completed ✅)

- Dataset: **Google Speech Commands v0.02** (subset: *yes, no, go, stop, up, down*)
- Feature extraction: **MFCC (Mel-Frequency Cepstral Coefficients)** with per-sample normalization
- Architecture:  
  `Conv2D → ReLU → MaxPool2D → Linear → ReLU → Linear`
- Optimizer: **Adam**, with **StepLR** learning rate scheduler  
- Training: 10 epochs

**Results**
| Metric | Accuracy |
|:-------|:----------|
| Training | ~98% |
| Validation | ~85% |
| Test | ~85% |

**Saved model:** `baseline_cnn_kws_vfinal.pt`

---

## ⚡ Phase 2: SNN Conversion (Completed ✅)

- Converted trained CNN → **Spiking Neural Network (SNN)** using **snnTorch**
- Replaced ReLU activations with **Leaky-Integrate-and-Fire (LIF)** neurons  
- Introduced **rate coding** (Poisson spike trains) for temporal input encoding
- Implemented **surrogate gradient (fast sigmoid)** for differentiable spiking dynamics
- Ran inference for multiple timesteps (`T=10–100`)

**Conversion workflow:**
1. Normalize CNN activations for stable spike propagation  
2. Copy convolutional + linear weights directly into SNN  
3. Simulate spikes over time using Poisson-coded inputs  
4. Accumulate temporal outputs for classification

---

## 🔋 Phase 3: Energy & Efficiency Evaluation (Completed ✅)

- Developed an **energy proxy metric** based on total spike activity:  
  \[
  \text{Energy Proxy} = \text{Average Spike Rate} \times T
  \]
- Computed **efficiency** as accuracy divided by energy cost
- Conducted parameter sweep over:
  - Leak constants **β ∈ {0.90, 0.95, 0.97, 0.99}**
  - Simulation timesteps **T ∈ {10, 25, 50, 75, 100}**
- Recorded:
  - Accuracy (Val/Test)
  - Spike Rate
  - Energy Proxy
  - Efficiency Score

**Key Insights**
- SNN achieves **~80–90% of CNN accuracy** with up to **5–10× lower energy consumption**.  
- Optimal performance around **β ≈ 0.95–0.97**, **T = 25–50**.  
- Demonstrated biologically inspired **accuracy–energy trade-off**.

---

## 📊 Phase 4: Visualization & Analysis (Completed ✅)

Generated several performance plots:
1. **Accuracy vs Timesteps (T)** — shows temporal convergence  
2. **Energy vs Timesteps (T)** — higher β increases energy cost  
3. **Efficiency vs β** — reveals best leak constant for energy-aware inference  
4. **Energy-Accuracy Scatter** — illustrates trade-off frontier between performance and power

Data and results are organized in a Pandas DataFrame for reproducibility and easy plotting.

---

## 🧾 Environment

| Library | Version |
|----------|----------|
| PyTorch | 2.8.0+cu126 |
| snnTorch | 0.9.1 |
| Torchaudio | 2.8.0+cu126 |
| NumPy | 1.26+ |
| tqdm | Latest |
| Platform | Google Colab (T4 GPU) / macOS M1 (local) |

---

## ✅ Completed Checklist

- [x] Setup PyTorch + snnTorch + Torchaudio environment  
- [x] Implemented MFCC preprocessing with normalization  
- [x] Trained baseline CNN with stable 85% accuracy  
- [x] Converted CNN → SNN (LIF neurons, surrogate gradients)  
- [x] Implemented rate coding (Poisson spike generation)  
- [x] Evaluated energy metrics and efficiency  
- [x] Generated comparative visualizations  
- [x] Documented all results in structured format  

---

## 🚀 Future Work

### 🔬 Phase 5 — Optimization & Extensions
- [ ] Fine-tune SNN using surrogate gradient training for better accuracy  
- [ ] Explore **temporal coding** (time-to-first-spike) as alternative input representation  
- [ ] Test deployment on neuromorphic simulators (Intel Loihi / SpiNNaker)  
- [ ] Extend dataset (more keywords, background noise augmentation)

### 📑 Phase 6 — Report & Presentation
- [ ] Prepare technical paper summarizing energy-accuracy results  
- [ ] Create slide deck and demo video illustrating spike activity  
- [ ] Include efficiency comparison with standard CNN  

---

## 📁 Repository Structure

```
Neuromorphic-KWS/
│                        # Speech Commands dataset (subset)
├── saved_models/
│   ├── baseline_cnn_kws_vfinal.pt # Trained CNN weights
│   └── snn_kws_model.pt           # Converted SNN (optional)
├── snn_conversion/
│   └── kws_cnn_to_snn.ipynb       # Full training + conversion notebook
├── baseline_cnn/
│   └── cnn.ipynb
└── README.md
```

---

## 💡 Summary

This project demonstrates how **a trained CNN for speech recognition can be transformed into a biologically inspired SNN** capable of event-driven, energy-efficient inference.  
The framework provides a reproducible baseline for future **neuromorphic machine learning** research, bridging conventional deep learning and spiking computation.
