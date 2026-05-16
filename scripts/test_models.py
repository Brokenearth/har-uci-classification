#!/usr/bin/env python3
"""Prueba de forward pass de LSTM y Conv1D+LSTM."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from har_classifier.models import Conv1DLSTMClassifier, LSTMClassifier
from har_classifier.training.trainer import Trainer


def main():
    b, c, t = 8, 9, 128
    x_ct = torch.randn(b, c, t)
    x_tc = x_ct.transpose(1, 2)

    lstm = LSTMClassifier()
    conv = Conv1DLSTMClassifier()

    out_lstm = lstm(x_tc)
    out_conv = conv(x_ct)
    out_lstm_via = Trainer._forward(lstm, x_ct)
    out_conv_via = Trainer._forward(conv, x_ct)

    assert out_lstm.shape == (b, 6), out_lstm.shape
    assert out_conv.shape == (b, 6), out_conv.shape
    assert out_lstm_via.shape == (b, 6)
    assert out_conv_via.shape == (b, 6)
    print("Forward pass OK: LSTM y Conv1D+LSTM")


if __name__ == "__main__":
    main()
