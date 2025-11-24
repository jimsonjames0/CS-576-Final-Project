#!/usr/bin/env python3
# live_audio_demo.py

import sounddevice as sd
import soundfile as sf
import numpy as np
import torch
import torchaudio
import torch.nn.functional as F
import torchaudio.transforms as T
from pathlib import Path
import matplotlib.pyplot as plt

SAMPLE_RATE = 16000
CLASSES = ["yes", "no", "go", "stop", "down", "up"]

# MFCC Transform (same config as training)
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


def record_until_enter(fs=SAMPLE_RATE, out_path="live_record.wav"):
    print("Recording... press ENTER to stop.")
    audio_list = []

    stream = sd.InputStream(
        samplerate=fs, channels=1, dtype="float32",
        callback=lambda indata, frames, time, status: audio_list.append(indata.copy())
    )
    stream.start()
    try:
        input()
    except KeyboardInterrupt:
        pass
    stream.stop()

    audio = np.concatenate(audio_list, axis=0)
    sf.write(out_path, audio, fs)
    print("Saved to", out_path)
    return Path(out_path)


TARGET_T = 100   # Time frames used during CNN training

def wav_to_mfcc(path):
    waveform, sr = sf.read(str(path))
    waveform = torch.tensor(waveform).float()

    # ensure [1, N]
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.ndim == 2:
        waveform = waveform.mean(dim=1, keepdim=True).transpose(0, 1)

    # resample
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

    # MFCC extraction
    mfcc = mfcc_transform(waveform).squeeze(0)  # [40, T]

    # normalize
    mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-6)
    mfcc = torch.clamp(mfcc, -2.0, 2.0)

    # enforce fixed time dimension
    T = mfcc.shape[1]
    if T > TARGET_T:
        mfcc = mfcc[:, :TARGET_T]
    elif T < TARGET_T:
        pad = TARGET_T - T
        mfcc = F.pad(mfcc, (0, pad))

    return waveform.squeeze(0), mfcc


def predict(cnn, snn, device, mfcc):
    mfcc_batch = mfcc.unsqueeze(0).to(device)

    with torch.no_grad():
        logits_cnn = cnn(mfcc_batch)
        pred_cnn = logits_cnn.argmax(1).item()

        out_TBC = snn(mfcc_batch)
        logits_snn = out_TBC.sum(0).squeeze(0)
        pred_snn = logits_snn.argmax().item()

    return pred_cnn, pred_snn


if __name__ == "__main__":
    cnn, snn, device = load_models()

    audio_path = record_until_enter()

    waveform, mfcc = wav_to_mfcc(audio_path)
    pred_cnn, pred_snn = predict(cnn, snn, device, mfcc)

    print("CNN Prediction:", CLASSES[pred_cnn])
    print("SNN Prediction:", CLASSES[pred_snn])