import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_KWS(nn.Module):
    """
    Clean standalone CNN architecture for Keyword Spotting inference.
    Compatible with baseline_cnn_kws_vfinal.pt.
    """

    def __init__(self, num_classes=6, flatten_dim=3840):
        super().__init__()
        self.flatten_dim = flatten_dim

        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.flatten_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # x: [B, 40, T]
        x = x.unsqueeze(1)        # → [B,1,40,T]
        x = self.features(x)
        x = torch.flatten(x, 1)   # → [B, F]

        # Match training feature dimension
        F_now = x.shape[1]
        if F_now > self.flatten_dim:
            x = x[:, :self.flatten_dim]
        elif F_now < self.flatten_dim:
            pad = self.flatten_dim - F_now
            x = F.pad(x, (0, pad))

        x = self.classifier(x)
        return x