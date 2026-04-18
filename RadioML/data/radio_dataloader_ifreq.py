"""
RadioML dataloader with instantaneous frequency feature.

Extends baseline with phase derivative channel:
  Input: [I, Q] (raw IQ)
  Feature: phase = atan2(Q, I), then compute phase derivative (frequency-related)
  Output: [I, Q, phase_derivative] = [3, T] instead of [2, T]

This allows testing if explicit frequency information improves WBFM classification.
"""

import os
import pickle
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler


def _load_rml2016a(pkl_path: str) -> Dict[Tuple[str, int], np.ndarray]:
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"RadioML file not found: {pkl_path}")
    with open(pkl_path, "rb") as f:
        try:
            data = pickle.load(f, encoding="latin1")
        except TypeError:
            data = pickle.load(f)
    if not isinstance(data, dict):
        raise ValueError("Invalid format: expected dict[(mod, snr)] -> ndarray")
    return data


def _pack_arrays(
    data_dict: Dict[Tuple[str, int], np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[int]]:
    mods = sorted({k[0] for k in data_dict.keys()})
    snrs = sorted({k[1] for k in data_dict.keys()})
    mod2idx = {m: i for i, m in enumerate(mods)}

    x_list, y_list, s_list = [], [], []
    for m in mods:
        for s in snrs:
            block = data_dict[(m, s)].astype(np.float32)  # [N, 2, 128]
            x_list.append(block)
            y_list.append(np.full((block.shape[0],), mod2idx[m], dtype=np.int64))
            s_list.append(np.full((block.shape[0],), s, dtype=np.int64))

    X = np.concatenate(x_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    snr = np.concatenate(s_list, axis=0)
    return X, y, snr, list(mods), list(snrs)


def _split_by_group(
    data_dict: Dict[Tuple[str, int], np.ndarray],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 1,
):
    rng = np.random.RandomState(seed)
    mods = sorted({k[0] for k in data_dict.keys()})
    snrs = sorted({k[1] for k in data_dict.keys()})

    offsets = {}
    cursor = 0
    for m in mods:
        for s in snrs:
            n = data_dict[(m, s)].shape[0]
            offsets[(m, s)] = (cursor, cursor + n)
            cursor += n

    train_idx, val_idx, test_idx = [], [], []
    for m in mods:
        for s in snrs:
            st, ed = offsets[(m, s)]
            idx = np.arange(st, ed)
            rng.shuffle(idx)

            n = len(idx)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)

            train_idx.append(idx[:n_train])
            val_idx.append(idx[n_train:n_train + n_val])
            test_idx.append(idx[n_train + n_val:])

    return np.concatenate(train_idx), np.concatenate(val_idx), np.concatenate(test_idx)


def _compute_instantaneous_frequency_channel(iq: np.ndarray) -> np.ndarray:
    """
    Compute phase derivative as a proxy for instantaneous frequency.
    
    Input: iq [2, T] where iq[0] = I (in-phase), iq[1] = Q (quadrature)
    Output: if_channel [T] phase derivative (unwrapped and normalized)
    
    Why this helps:
    - WBFM has time-varying frequency (phase derivative changes over time)
    - AM-DSB has static frequency (phase stays roughly constant)
    - Adding this explicit feature lets the model capture frequency modulation
    """
    i = iq[0]  # in-phase channel
    q = iq[1]  # quadrature channel
    
    # Compute phase: atan2(Q, I)
    phase = np.arctan2(q, i)
    
    # Unwrap phase (continuous despite wrapping at ±π)
    phase_unwrapped = np.unwrap(phase)
    
    # Compute phase derivative (frequency-related, normalized)
    phase_diff = np.diff(phase_unwrapped, prepend=phase_unwrapped[0])
    
    # Normalize: scale to approximately [-1, 1]
    max_val = np.max(np.abs(phase_diff))
    if max_val > 1e-6:
        phase_diff = phase_diff / (max_val + 1e-6)
    
    return phase_diff


class RadioML2016aDatasetIFreq(Dataset):
    """
    RadioML dataset with instantaneous frequency feature.
    
    Output shape: [3, 128] instead of [2, 128]
      Channel 0: I (in-phase)
      Channel 1: Q (quadrature)
      Channel 2: Phase derivative (instantaneous frequency)
    """
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        snr: np.ndarray,
        indices: np.ndarray,
        normalize: bool = True,
        return_snr: bool = False,
    ):
        self.X = X[indices].copy()  # [N, 2, 128]
        self.y = y[indices].copy()
        self.snr = snr[indices].copy()
        self.return_snr = return_snr

        # Add instantaneous frequency channel: [N, 2, 128] -> [N, 3, 128]
        n_samples = self.X.shape[0]
        if_channel = np.zeros((n_samples, 128), dtype=np.float32)
        for i in range(n_samples):
            if_channel[i] = _compute_instantaneous_frequency_channel(self.X[i])
        
        self.X = np.concatenate([self.X, if_channel[:, np.newaxis, :]], axis=1)  # [N, 3, 128]

        # Normalize all channels
        if normalize:
            max_abs = np.max(np.abs(self.X), axis=(1, 2), keepdims=True)
            max_abs = np.maximum(max_abs, 1e-6)
            self.X = self.X / max_abs

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx]).float()  # [3, 128]
        y = torch.tensor(self.y[idx], dtype=torch.long)

        if self.return_snr:
            s = torch.tensor(self.snr[idx], dtype=torch.long)
            return x, y, s
        return x, y


def get_radioml2016a_dataloaders_ifreq(
    pkl_path: str,
    batch_size: int = 256,
    test_batch_size: Optional[int] = None,
    num_workers: int = 4,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 1,
    normalize: bool = True,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    pin_memory: bool = True,
):
    """Load dataloaders with instantaneous frequency feature."""
    
    data_dict = _load_rml2016a(pkl_path)
    X, y, snr, mods, snrs = _pack_arrays(data_dict)
    train_idx, val_idx, test_idx = _split_by_group(
        data_dict, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed
    )

    train_ds = RadioML2016aDatasetIFreq(X, y, snr, train_idx, normalize=normalize)
    val_ds = RadioML2016aDatasetIFreq(X, y, snr, val_idx, normalize=normalize)
    test_ds = RadioML2016aDatasetIFreq(X, y, snr, test_idx, normalize=normalize, return_snr=True)

    train_sampler = None
    if distributed:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=seed
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=test_batch_size or batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=test_batch_size or batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader
