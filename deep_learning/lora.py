import torch
import torch.nn as nn


class LinearLora(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float = 1.0, bias: bool = False):
        super().__init__()

        self.alpha = alpha
        self.rank = rank
        self.alpha_scale = alpha / rank

        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.lora_a = nn.Linear(in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, out_features, bias=False)

        for param in self.linear.parameters():
            param.requires_grad = False

        nn.init.kaiming_normal_(self.lora_a.weight)
        nn.init.zeros_(self.lora_b.weight)

        self.merged = False

    def merge(self):
        if self.merged:
            return
        # W' = W + B @ A * (alpha / rank)
        self.linear.weight.data += (self.lora_b.weight @ self.lora_a.weight) * self.alpha_scale
        self.merged = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, in_features]
        Returns:
            [batch_size, out_features]
        """
        if self.merged:
            return self.linear(x)
        return self.linear(x) + self.lora_b(self.lora_a(x)) * self.alpha_scale


if __name__ == '__main__':
    batch_size, in_features, out_features, rank, alpha = 4, 768, 768, 8, 16.0

    model = LinearLora(in_features, out_features, rank=rank, alpha=alpha)
    x = torch.randn(batch_size, in_features)

    out_before_merge = model(x)
    print(out_before_merge.shape)

    model.merge()
    out_after_merge = model(x)
    print(out_after_merge.shape)

    diff = (out_before_merge - out_after_merge).abs().max().item()
    print("Max diff: ", diff)
