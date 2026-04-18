"""
RadioML CNN trained with instantaneous frequency features.

Minimal change from baseline:
  - Input channels: 2 (I, Q) → 3 (I, Q, phase_derivative)
  - First Conv1d layer: in_channels 2 → 3
  - All other architecture identical to baseline
"""

import argparse
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from radio_dataloader_ifreq import get_radioml2016a_dataloaders_ifreq


class RadioMLCNNIFreq(nn.Module):
    """1D CNN for RadioML with instantaneous frequency feature (3 channels)."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=7, stride=1, padding=3, bias=False),  # 3 -> 64
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 128 -> 64
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 64 -> 32
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.4),
            nn.Linear(8192, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.head(x)
        return x
