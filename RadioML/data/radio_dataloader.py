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
            data = pickle.load(f, encoding="latin1")  # python2 pickle兼容
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


class RadioML2016aDataset(Dataset):
    """
    默认返回:
      x: FloatTensor [2, 128]
      y: LongTensor  [1]
    可选:
      as_2d=True  -> x: [1, 2, 128] (适配2D卷积)
      return_snr=True -> 返回 (x, y, snr)
    """
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        snr: np.ndarray,
        indices: np.ndarray,
        normalize: bool = True,
        as_2d: bool = False,
        return_snr: bool = False,
    ):
        self.X = X[indices].copy()
        self.y = y[indices].copy()
        self.snr = snr[indices].copy()
        self.as_2d = as_2d
        self.return_snr = return_snr

        if normalize:
            max_abs = np.max(np.abs(self.X), axis=(1, 2), keepdims=True)
            max_abs = np.maximum(max_abs, 1e-6)
            self.X = self.X / max_abs

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx]).float()  # [2,128]
        if self.as_2d:
            x = x.unsqueeze(0)  # [1,2,128]
        y = torch.tensor(self.y[idx], dtype=torch.long)

        if self.return_snr:
            s = torch.tensor(self.snr[idx], dtype=torch.long)
            return x, y, s
        return x, y


def get_radioml2016a_dataloaders(
    pkl_path: str,
    batch_size: int = 256,
    test_batch_size: Optional[int] = None,
    num_workers: int = 4,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 1,
    normalize: bool = True,
    as_2d: bool = False,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    pin_memory: bool = True,
):
    data_dict = _load_rml2016a(pkl_path)
    X, y, snr, mods, snrs = _pack_arrays(data_dict)
    train_idx, val_idx, test_idx = _split_by_group(
        data_dict, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed
    )

    train_ds = RadioML2016aDataset(X, y, snr, train_idx, normalize=normalize, as_2d=as_2d)
    val_ds = RadioML2016aDataset(X, y, snr, val_idx, normalize=normalize, as_2d=as_2d)
    test_ds = RadioML2016aDataset(X, y, snr, test_idx, normalize=normalize, as_2d=as_2d)

    train_sampler = None
    val_sampler = None
    test_sampler = None
    if distributed:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True
        )
        val_sampler = DistributedSampler(
            val_ds, num_replicas=world_size, rank=rank, shuffle=False
        )
        test_sampler = DistributedSampler(
            test_ds, num_replicas=world_size, rank=rank, shuffle=False
        )

    if test_batch_size is None:
        test_batch_size = batch_size

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=test_batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=test_batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    meta = {
        "mods": mods,
        "snrs": snrs,
        "num_classes": len(mods),
        "input_shape": [2, 128] if not as_2d else [1, 2, 128],
    }
    return train_loader, val_loader, test_loader, train_sampler, meta
