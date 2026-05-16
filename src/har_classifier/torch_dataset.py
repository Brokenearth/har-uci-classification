"""Dataset PyTorch para ventanas HAR."""

import numpy as np
import torch
from torch.utils.data import Dataset


class HARDataset(Dataset):
    """Entrega tensores (C, T) float32 para Conv1D/LSTM."""

    def __init__(self, X: np.ndarray, y: np.ndarray, layout: str = "channels_first"):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.layout = layout

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx]
        if self.layout == "channels_first":
            x = x.transpose(1, 0)
        return torch.from_numpy(x), torch.tensor(self.y[idx], dtype=torch.long)
