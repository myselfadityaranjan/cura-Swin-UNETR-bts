import torch

def pad_to_length(tensor: torch.Tensor, length: int) -> torch.Tensor: #force 3 elements if not possible
    if tensor.numel() < length:
        needed = length - tensor.numel()
        pad = torch.zeros(needed, device=tensor.device, dtype=tensor.dtype)
        tensor = torch.cat([tensor, pad], dim=0)
    return tensor
