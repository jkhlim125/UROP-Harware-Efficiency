"""
RadioML refined two-branch CNN (v3).

Refinements over v2:
1. Keep the successful two-branch late-fusion structure
2. Keep the smoothed IF idea
3. Expand IF input to a medium-size interpretable feature set
4. Remove v2 class weighting and return to standard cross-entropy

Architecture:
- IQ branch: same as v2
- IF branch: same lightweight v2 stack, now with 8-channel input
- Fusion: concatenate + classify
"""

import argparse
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from radio_dataloader_branch_v3 import get_radioml2016a_dataloaders_branch_v3


class IQFeatureExtractorV3(nn.Module):
    """Feature extractor for raw IQ branch (same as v2)."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class IFFeatureExtractorV3(nn.Module):
    """IF feature extractor with the same v2 depth and an 8-channel input."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(8, 32, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class RadioMLBranchCNNV3(nn.Module):
    """Refined two-branch CNN with minimal v3 IF-feature expansion."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.iq_branch = IQFeatureExtractorV3()
        self.if_branch = IFFeatureExtractorV3()
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(8192 + 4096, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(512, num_classes),
        )

    def forward(self, iq: torch.Tensor, if_feat: torch.Tensor) -> torch.Tensor:
        iq_features = self.iq_branch(iq)
        if_features = self.if_branch(if_feat)
        fused = torch.cat([iq_features, if_features], dim=1)
        logits = self.classifier(fused)
        return logits
