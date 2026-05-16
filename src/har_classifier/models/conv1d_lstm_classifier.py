"""Clasificador Conv1D + LSTM según arquitectura del plan."""

import torch
import torch.nn as nn

from har_classifier.config import (
    CONV_FILTERS,
    CONV_KERNEL_SIZE,
    DROPOUT,
    HIDDEN_SIZE,
    NUM_CHANNELS,
    NUM_CLASSES,
    NUM_LSTM_LAYERS,
)


class Conv1DLSTMClassifier(nn.Module):
    """Conv1d -> BatchNorm -> ReLU -> MaxPool (x2) -> LSTM -> Linear."""

    def __init__(
        self,
        input_channels: int = NUM_CHANNELS,
        conv_filters: tuple[int, int] = CONV_FILTERS,
        kernel_size: int = CONV_KERNEL_SIZE,
        hidden_size: int = HIDDEN_SIZE,
        num_layers: int = NUM_LSTM_LAYERS,
        num_classes: int = NUM_CLASSES,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        f1, f2 = conv_filters
        pad = kernel_size // 2
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(input_channels, f1, kernel_size, padding=pad),
            nn.BatchNorm1d(f1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(f1, f2, kernel_size, padding=pad),
            nn.BatchNorm1d(f2),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(
            input_size=f2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, seq) o (batch, seq, channels)
        if x.dim() == 3 and x.shape[1] != NUM_CHANNELS:
            x = x.transpose(1, 2)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x.transpose(1, 2)
        _, (h_n, _) = self.lstm(x)
        out = self.dropout(h_n[-1])
        return self.fc(out)
