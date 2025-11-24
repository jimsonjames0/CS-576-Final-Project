import torch
import torchaudio
import torchaudio.transforms as T
import soundfile as sf
from pathlib import Path

SAMPLE_RATE = 16000
N_MFCC = 40

# MFCC transform (same as training)
mfcc_transform = T.MFCC(
    sample_rate=SAMPLE_RATE,
    n_mfcc=N_MFCC,
    melkwargs={
        "n_fft": 400,
        "hop_length": 160,
        "n_mels": 40,
        "center": False,
    },
)


def load_audio(path: Path):
    """
    Load WAV using soundfile, resample to 16 kHz, return tensor [N].
    """
    waveform_np, sr = sf.read(str(path))
    waveform = torch.tensor(waveform_np).float()

    # Ensure mono
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=1)

    waveform = waveform.unsqueeze(0)  # [1, N]

    # Resample if needed
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

    return waveform.squeeze(0)


def wav_to_mfcc(path: Path):
    """
    Returns: (waveform, mfcc) where:
        waveform = [N]
        mfcc = [40, T]
    """
    waveform = load_audio(path)
    wav_2d = waveform.unsqueeze(0)  # shape [1, N]

    mfcc = mfcc_transform(wav_2d).squeeze(0)  # [40, T]

    # per-sample normalization
    mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-6)
    mfcc = torch.clamp(mfcc, -2.0, 2.0)

    return waveform, mfcc