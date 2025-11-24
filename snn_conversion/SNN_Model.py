import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate

# Surrogate gradient for SNN spiking
spike_grad = surrogate.fast_sigmoid()


class SNN_KWS(nn.Module):
    """
    Clean, inference-only SNN model used for real-time audio demo.
    Converts CNN_KWS into a spiking model with LIF neurons.
    """

    def __init__(self, base_cnn, num_steps=50, beta=0.95):
        super().__init__()

        self.num_steps = num_steps

        # keep a handle to CNN feature extractor and flatten_dim
        self.features = base_cnn.features
        self.fc1 = base_cnn.classifier[0]   # Linear -> 64
        self.fc2 = base_cnn.classifier[2]   # Linear -> 6
        self.flatten_dim = base_cnn.flatten_dim  # match CNN padding logic

        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x):
        """
        x: [B, 40, T]
        Returns:
            spikes[T, B, 6]
        """
        spk_out_list = []

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        x = x.unsqueeze(1)        # [B,1,40,T]

        for _ in range(self.num_steps):
            h = self.features(x)          # [B,C,H,W]
            h = torch.flatten(h, 1)       # [B,F]

            # ⭐ CRITICAL: enforce same flatten_dim as CNN_KWS
            F_now = h.shape[1]
            if F_now > self.flatten_dim:
                h = h[:, :self.flatten_dim]
            elif F_now < self.flatten_dim:
                pad = self.flatten_dim - F_now
                h = F.pad(h, (0, pad))

            h = F.relu(self.fc1(h))       # → 64 dim
            spk1, mem1 = self.lif1(h, mem1)

            h2 = self.fc2(spk1)          # → 6 logits
            spk2, mem2 = self.lif2(h2, mem2)

            spk_out_list.append(spk2)

        return torch.stack(spk_out_list)  # [T,B,6]