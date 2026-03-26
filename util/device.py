"""
Device utility for cross-platform GPU support (CUDA, MPS, CPU).

Supports:
- NVIDIA CUDA GPUs
- Apple Metal Performance Shaders (MPS) on M1/M2/M3
- CPU fallback
"""

import torch


def select_device(prefer: str = "auto") -> torch.device:
    """
    Select the appropriate device based on availability and user preference.
    
    Args:
        prefer (str): Device preference. Options:
            - "auto": Automatically select the best available device
            - "cuda": Force CUDA (raises error if not available)
            - "mps": Force MPS (raises error if not available)
            - "cpu": Force CPU
    
    Returns:
        torch.device: The selected device
    
    Raises:
        RuntimeError: If requested device is not available
    
    Examples:
        >>> device = select_device("auto")
        >>> device = select_device("cuda")
        >>> model.to(device)
    """
    if prefer == "cuda":
        if torch.cuda.is_available():
            print("✓ Using CUDA device")
            return torch.device("cuda")
        raise RuntimeError("CUDA requested but not available.")
    
    if prefer == "mps":
        if torch.backends.mps.is_available():
            print("✓ Using MPS (Metal Performance Shaders) device")
            return torch.device("mps")
        raise RuntimeError("MPS requested but not available.")
    
    if prefer == "cpu":
        print("✓ Using CPU device")
        return torch.device("cpu")
    
    # auto mode
    if torch.cuda.is_available():
        print("✓ Using CUDA device (auto-selected)")
        return torch.device("cuda")
    
    if torch.backends.mps.is_available():
        print("✓ Using MPS device (auto-selected)")
        return torch.device("mps")
    
    print("✓ Using CPU device (fallback)")
    return torch.device("cpu")


def get_autocast_context(device: torch.device, enabled: bool = True):
    """
    Get the appropriate autocast context for the device.
    
    Args:
        device (torch.device): The device to optimize for
        enabled (bool): Whether autocast is enabled
    
    Returns:
        torch.autocast: Autocast context manager
    
    Examples:
        >>> device = select_device("auto")
        >>> with get_autocast_context(device):
        ...     output = model(input)
    """
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", enabled=enabled)
    elif device.type == "mps":
        return torch.autocast(device_type="mps", enabled=enabled)
    else:
        return torch.autocast(device_type="cpu", enabled=False)


def get_grad_scaler(device: torch.device):
    """
    Get GradScaler if using CUDA mixed precision, else None.
    
    Args:
        device (torch.device): The device
    
    Returns:
        torch.cuda.amp.GradScaler or None
    
    Examples:
        >>> device = select_device("auto")
        >>> scaler = get_grad_scaler(device)
        >>> if scaler:
        ...     scaler.scale(loss).backward()
        ...     scaler.step(optimizer)
        ...     scaler.update()
    """
    if device.type == "cuda":
        return torch.cuda.amp.GradScaler()
    return None
