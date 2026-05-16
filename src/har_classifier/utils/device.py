"""Detección de dispositivo CUDA / CPU."""

import torch


def get_device(verbose: bool = True) -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            print(f"Usando GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        if verbose:
            print("CUDA no disponible; usando CPU.")
    return device
