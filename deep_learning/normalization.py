import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x)
        return output * self.weight


class AdaptiveLayerNorm(nn.Module):
    def __init__(self, dim: int, cond_dim: int = None, eps: float = 1e-8):
        super().__init__()
        self.dim = dim
        self.eps = eps
        
        cond_dim = cond_dim if cond_dim is not None else dim

        self.linear = nn.Linear(cond_dim, 2 * dim, bias=False)
        
        nn.init.zeros_(self.linear.weight)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        return (x - mean) * torch.rsqrt(var + self.eps)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, ..., dim]
            condition: [batch, ..., cond_dim]
        """
        x_norm = self._norm(x)
        
        style = self.linear(condition)
        gamma, beta = style.chunk(2, dim=-1)
        
        return x_norm * (1.0 + gamma) + beta
