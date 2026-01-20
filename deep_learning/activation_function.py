import torch
import torch.nn as nn


class ReLU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, min=0)

class Sigmoid(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1 / (1 + e^(-x))
        return 1 / (1 + torch.exp(-x))


class Softmax(nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        max_values, _ = torch.max(x, dim=self.dim, keepdim=True)
        x_stabilized = x - max_values

        x_exp = torch.exp(x_stabilized)
        return x_exp / torch.sum(x_exp, dim=self.dim, keepdim=True)


class SiLU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)
