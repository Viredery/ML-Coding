import torch
import math


class PositionalEncoding(torch.nn.Module):
    def __init__(self, max_seq_len: int, d_model: int):
        super().__init__()

        self.max_seq_len = max_seq_len
        self.d_model = d_model

        if self.d_model % 2 != 0:
            raise ValueError(f'd_model must be even, got {self.d_model}')

        positional_encoding = torch.zeros(self.max_seq_len, self.d_model)

        # [seq_len, 1]
        position = torch.arange(0, self.max_seq_len, dtype=torch.float32).unsqueeze(1)

        # 10000 ^ (2i / d_model) -> exp( 2i / d_model * -log(10000))
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32) * 
            -(math.log(10000.0) / self.d_model)
        )

        sin = torch.sin(position * div_term)
        cos = torch.cos(position * div_term)

        positional_encoding[:, 0::2] = sin
        positional_encoding[:, 1::2] = cos

        self.register_buffer('positional_encoding', positional_encoding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, hidden_size]
        """
        return x + self.positional_encoding[:x.size(1), :]


if __name__ == "__main__":
    max_seq_len = 100
    d_model = 512
    batch_size = 4
    seq_len = 60

    pos_encoder = PositionalEncoding(max_seq_len, d_model)
    x = torch.randn(batch_size, seq_len, d_model)
    output = pos_encoder(x)
    print(output.shape)
