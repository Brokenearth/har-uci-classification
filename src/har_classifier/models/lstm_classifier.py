"""Clasificador LSTM para series temporales multisensoriales."""

import torch
import torch.nn as nn

from har_classifier.config import DROPOUT, HIDDEN_SIZE, NUM_CHANNELS, NUM_CLASSES, NUM_LSTM_LAYERS


class LSTMClassifier(nn.Module):
    """LSTM + capa lineal para clasificación de actividades."""

    def __init__(
        self,
        input_size: int = NUM_CHANNELS,
        hidden_size: int = HIDDEN_SIZE,
        num_layers: int = NUM_LSTM_LAYERS,
        num_classes: int = NUM_CLASSES,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, channels)
        out, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]
        out = self.dropout(last_hidden)
        return self.fc(out)
