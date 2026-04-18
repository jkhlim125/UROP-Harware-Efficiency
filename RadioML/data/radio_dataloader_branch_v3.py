"""
RadioML dataloader for refined two-branch experiment (v3).

Improvements over v2:
- Keep the core instantaneous-frequency signal
- Expand the IF branch input to a medium-size interpretable feature set
- Preserve the same split/loading behavior as earlier branch experiments

Returns:
- raw_iq: [2, 128]
- if_feat: [8, 128]
- label
- snr (for analysis)
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
            block = data_dict[(m, s)].astype(np.float32)
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


def _smooth_moving_average(signal: np.ndarray, window_size: int = 3) -> np.ndarray:
    """Simple moving average smoothing."""
    if window_size <= 1:
        return signal

    kernel = np.ones(window_size, dtype=np.float32) / float(window_size)
    padded = np.pad(signal, (window_size // 2, window_size // 2), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed[:len(signal)]


def _compute_if_feature_v3(
    iq: np.ndarray,
    smooth_window: int = 3,
) -> np.ndarray:
    """
    Build a medium-size interpretable IF feature tensor for the branch model.

    Channels:
    1. IF(t)
    2. IF_smoothed(t)
    3. IF_abs(t)
    4. IF_energy (broadcast)
    5. IF_variance (broadcast)
    6. amplitude_envelope(t)
    7. amplitude_mean (broadcast)
    8. amplitude_variance (broadcast)
    """
    i = iq[0]
    q = iq[1]

    phase = np.arctan2(q, i)
    phase_unwrapped = np.unwrap(phase)
    if_signal = np.diff(phase_unwrapped, prepend=phase_unwrapped[0]).astype(np.float32)
    if_smoothed = _smooth_moving_average(if_signal, window_size=smooth_window).astype(np.float32)
    if_abs = np.abs(if_signal).astype(np.float32)

    if_energy_scalar = np.mean(if_signal ** 2, dtype=np.float32)
    if_variance_scalar = np.var(if_signal, dtype=np.float32)

    amplitude_envelope = np.sqrt(i ** 2 + q ** 2).astype(np.float32)
    amplitude_mean_scalar = np.mean(amplitude_envelope, dtype=np.float32)
    amplitude_variance_scalar = np.var(amplitude_envelope, dtype=np.float32)

    phase_scale = np.max(np.abs(if_signal))
    phase_scale = max(float(phase_scale), 1e-6)
    amp_scale = np.max(amplitude_envelope)
    amp_scale = max(float(amp_scale), 1e-6)

    features = np.stack(
        [
            if_signal / phase_scale,
            if_smoothed / phase_scale,
            if_abs / phase_scale,
            np.full_like(if_signal, if_energy_scalar / (phase_scale ** 2)),
            np.full_like(if_signal, if_variance_scalar / (phase_scale ** 2)),
            amplitude_envelope / amp_scale,
            np.full_like(amplitude_envelope, amplitude_mean_scalar / amp_scale),
            np.full_like(amplitude_envelope, amplitude_variance_scalar / (amp_scale ** 2)),
        ],
        axis=0,
    )
    return features.astype(np.float32)


class RadioML2016aDatasetBranchV3(Dataset):
    """
    RadioML dataset for refined two-branch architecture (v3).

    Returns:
    - raw_iq: [2, 128]
    - if_feat: [8, 128]
    - label
    - snr (optional)
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
        self.X = X[indices].copy()
        self.y = y[indices].copy()
        self.snr = snr[indices].copy()
        self.return_snr = return_snr

        n_samples = self.X.shape[0]
        self.if_feats = np.zeros((n_samples, 8, 128), dtype=np.float32)
        for i in range(n_samples):
            self.if_feats[i] = _compute_if_feature_v3(self.X[i], smooth_window=3)

        if normalize:
            max_abs = np.max(np.abs(self.X), axis=(1, 2), keepdims=True)
            max_abs = np.maximum(max_abs, 1e-6)
            self.X = self.X / max_abs

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        raw_iq = torch.from_numpy(self.X[idx]).float()
        if_feat = torch.from_numpy(self.if_feats[idx]).float()
        y = torch.tensor(self.y[idx], dtype=torch.long)

        if self.return_snr:
            s = torch.tensor(self.snr[idx], dtype=torch.long)
            return raw_iq, if_feat, y, s
        return raw_iq, if_feat, y


def get_radioml2016a_dataloaders_branch_v3(
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
    """Load dataloaders for refined two-branch architecture (v3)."""

    data_dict = _load_rml2016a(pkl_path)
    X, y, snr, mods, snrs = _pack_arrays(data_dict)
    train_idx, val_idx, test_idx = _split_by_group(
        data_dict, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed
    )

    train_ds = RadioML2016aDatasetBranchV3(X, y, snr, train_idx, normalize=normalize)
    val_ds = RadioML2016aDatasetBranchV3(X, y, snr, val_idx, normalize=normalize)
    test_ds = RadioML2016aDatasetBranchV3(
        X, y, snr, test_idx, normalize=normalize, return_snr=True
    )

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
