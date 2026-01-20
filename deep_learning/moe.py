import torch
import torch.nn as nn
import torch.nn.functional as F

class MoELayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_experts: int, top_k: int = 2):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.top_k = top_k

        self.experts = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(num_experts)])
        self.gate = nn.Linear(input_dim, num_experts)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, input_dim]
        """
        router_logits = self.gate(x) # [batch_size, seq_len, num_experts]

        if self.top_k < self.num_experts:
            topk_logits, topk_indices = torch.topk(router_logits, self.top_k, dim=-1)
            
            mask = torch.full_like(router_logits, float('-inf'))
            mask = mask.scatter(2, topk_indices, topk_logits)
            router_logits = mask

        gate_score = F.softmax(router_logits, dim=-1) # [batch_size, seq_len, num_experts]

        expert_out = torch.stack([expert(x) for expert in self.experts], dim=2) # [B, L, E, out]

        out = (expert_out * gate_score.unsqueeze(-1)).sum(dim=2)
        
        return out

if __name__ == "__main__":
    # Test
    batch_size, seq_len, dim = 2, 10, 32
    num_experts = 4
    top_k = 2
    
    x = torch.randn(batch_size, seq_len, dim)
    moe = MoELayer(dim, dim, num_experts, top_k=top_k)
    
    out = moe(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
