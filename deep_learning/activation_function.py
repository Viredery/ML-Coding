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


class SwiGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        # Swish(SiLU) + GLU
        self.dim_in = dim_in
        self.dim_out = dim_out
        
        self.linear = nn.Linear(dim_in, dim_out * 2)
        self.silu = SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        gate, value = x.chunk(2, dim=-1)
        return self.silu(gate) * value
