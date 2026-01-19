import torch


class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, max_seq_len: int, head_dim: int, theta: float = 10000.0):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim

        if self.head_dim % 2 != 0:
            raise ValueError(f'head_dim must be even, got {self.head_dim}')

        # [head_dim / 2,]
        inv_freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        # [max_seq_len]
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        # [max_seq_len, head_dim / 2]
        freqs = torch.outer(positions, inv_freqs)

        cos_cached = torch.cos(freqs).unsqueeze(0)
        sin_cached = torch.sin(freqs).unsqueeze(0)

        self.register_buffer('cos_cached', cos_cached)
        self.register_buffer('sin_cached', sin_cached)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, hidden_size]
        """
        ori_type = x.dtype
        batch_size, seq_len, hidden_size = x.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f'seq_len must be less than or equal to max_seq_len, got {seq_len} and {self.max_seq_len}')

        cos = self.cos_cached[:, :seq_len, :]
        sin = self.sin_cached[:, :seq_len, :]

        x = x.to(torch.float32)
        d_half = hidden_size // 2
        # [batch_size, seq_len, d_half]
        x1, x2 = x.split(d_half, dim=-1)

        res = torch.empty_like(x)
        res[..., :d_half] = x1 * cos - x2 * sin
        res[..., d_half:] = x1 * sin + x2 * cos

        return res.to(ori_type)


if __name__ == "__main__":
    max_seq_len = 100
    head_dim = 512
    batch_size = 4
    seq_len = 60

    rotary_positional_embedding = RotaryPositionalEmbedding(max_seq_len, head_dim)
    x = torch.randn(batch_size, seq_len, head_dim)
    output = rotary_positional_embedding(x)
    print(output.shape)
