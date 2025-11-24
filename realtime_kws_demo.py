#!/usr/bin/env python3
# Real-Time Keyword Spotting using CNN + SNN (Fixed Version)

import sounddevice as sd
import soundfile as sf
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
from collections import deque
from pathlib import Path

SAMPLE_RATE = 16000
WINDOW_DURATION = 0.75
WINDOW_SIZE = int(SAMPLE_RATE * WINDOW_DURATION)

CLASSES = ["yes", "no", "go", "stop", "down", "up"]
TARGET_T = 100

mfcc_transform = T.MFCC(
    sample_rate=SAMPLE_RATE,
    n_mfcc=40,
    melkwargs={
        "n_fft": 400,
        "hop_length": 160,
        "n_mels": 40,
        "center": False,
    },
)

def load_models():
    from baseline_cnn.CNN_Model import CNN_KWS
    from snn_conversion.SNN_Model import SNN_KWS

    device = torch.device("cpu")

    cnn_path = Path("saved_models/baseline_cnn_kws_vfinal.pt")
    snn_path = Path("saved_models/snn_kws_beta0.95_T50.pt")

    cnn = CNN_KWS(num_classes=6, flatten_dim=3840).to(device)
    cnn.load_state_dict(torch.load(cnn_path, map_location=device))
    cnn.eval()

    snn = SNN_KWS(cnn, num_steps=50, beta=0.95).to(device)
    snn.load_state_dict(torch.load(snn_path, map_location=device))
    snn.eval()

    return cnn, snn, device

def compute_mfcc(waveform):
    waveform = torch.tensor(waveform).float().unsqueeze(0)
    mfcc = mfcc_transform(waveform).squeeze(0)

    mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-6)
    mfcc = torch.clamp(mfcc, -2.0, 2.0)

    T_cur = mfcc.shape[1]
    if T_cur > TARGET_T:
        mfcc = mfcc[:, :TARGET_T]
    else:
        mfcc = F.pad(mfcc, (0, TARGET_T - T_cur))

    return mfcc

def run_inference(cnn, snn, device, mfcc):
    mfcc_batch = mfcc.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = cnn(mfcc_batch)
        pred_cnn = logits.argmax(1).item()

        out_TBC = snn(mfcc_batch)
        spikes = (out_TBC > 0).float().sum().item()
        logits_snn = out_TBC.sum(0).squeeze(0)
        pred_snn = logits_snn.argmax().item()

    return pred_cnn, pred_snn, spikes, logits

def realtime_demo():
    print("Loading models...")
    cnn, snn, device = load_models()
    print("Models loaded.\n")

    print("Starting microphone stream ...")
    print("Speak freely. Keywords: yes, no, go, stop, down, up")
    print("Press CTRL+C to exit.\n")

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 4))
    spike_window = deque(maxlen=60)

    line, = ax.plot([])
    ax.set_title("SNN Spike Activity (Live)")
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Spike Count")

    CHUNK = 1024
    OVERLAP = 6
    REQUIRED = WINDOW_SIZE // OVERLAP

    buffer = np.zeros((0, 1), dtype=np.float32)

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32") as stream:
        while True:
            try:
                chunk, _ = stream.read(CHUNK)
                buffer = np.concatenate([buffer, chunk], axis=0)

                if buffer.shape[0] >= REQUIRED:
                    start = max(0, buffer.shape[0] - REQUIRED)
                    window = buffer[start:].copy()

                    mfcc = compute_mfcc(window.squeeze())
                    pred_cnn, pred_snn, spike_count, logits = run_inference(cnn, snn, device, mfcc)

                    conf = torch.softmax(logits, dim=1).max().item()

                    if conf < 0.55:
                        label = "none"
                    else:
                        label = CLASSES[pred_snn]

                    print(f"Detected: {label:<5} | Conf: {conf:.2f} | Spikes: {spike_count}")

                    spike_window.append(spike_count)

                    values = list(spike_window)
                    line.set_ydata(values)
                    line.set_xdata(range(len(values)))

                    if len(values) > 1:
                        ymin = max(0, min(values) - 5)
                        ymax = max(values) + 5
                        ax.set_ylim(ymin, ymax)

                    ax.set_xlim(0, len(values))

                    fig.canvas.draw()
                    fig.canvas.flush_events()

            except KeyboardInterrupt:
                print("\nStopping real-time KWS...")
                break

if __name__ == "__main__":
    realtime_demo()
