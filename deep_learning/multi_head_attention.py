import math
import torch
from torch import nn

class MultiHeadAttention(nn.Module):
	def __init__(self, input_dim: int, dim: int, num_head: int, dropout: float = 0.0):
		super().__init__()
		self.input_dim = input_dim
		self.dim = dim
		self.num_head = num_head

		assert self.dim > 0

		self.to_q = nn.Linear(input_dim, dim * num_head, bias=False)
		self.to_k = nn.Linear(input_dim, dim * num_head, bias=False)
		self.to_v = nn.Linear(input_dim, dim * num_head, bias=False)
		self.to_out = nn.Linear(dim * num_head, input_dim, bias=False)

		self.softmax = nn.Softmax(dim=-1)
		self.dropout = nn.Dropout(dropout)

	def forward(self, 
	            hidden_state: torch.Tensor,
	            attention_mask: torch.Tensor = None,
	            kv_cache: tuple[torch.Tensor, torch.Tensor] = None):
		"""
		Args:
			hidden_state: [batch_size, seq_len, input_dim]
			attention_mask: [batch_size, seq_len, seq_len]
			kv_cache: tuple of (k_cache, v_cache), each of shape [batch_size, num_head, past_seq_len, dim]
		"""

		batch_size, seq_len = hidden_state.shape[:2]

		scale = 1 / math.sqrt(self.dim)

		q = self.to_q(hidden_state).view(batch_size, seq_len, self.num_head, self.dim).transpose(1, 2)
		k = self.to_k(hidden_state).view(batch_size, seq_len, self.num_head, self.dim).transpose(1, 2)
		v = self.to_v(hidden_state).view(batch_size, seq_len, self.num_head, self.dim).transpose(1, 2)

		if kv_cache is not None:
			k_cache, v_cache = kv_cache
			k = torch.cat([k_cache, k], dim=2)
			v = torch.cat([v_cache, v], dim=2)

		new_kv_cache = (k, v)

		ori_type = q.dtype

		attn_logits = q @ k.transpose(-2, -1)
		attn_logits *= scale
		if attention_mask is not None:
			attn_logits = attn_logits.masked_fill(attention_mask[:, None, :, :] == 0, float("-inf"))
		attn = self.softmax(attn_logits.to(torch.float32)).to(ori_type)
		attn = self.dropout(attn)

		out = attn @ v

		out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.num_head * self.dim)

		out = self.to_out(out)

		return out, new_kv_cache


if __name__ == "__main__":
	# Test normal attention
	hidden_state = torch.randn(2, 10, 1024)
	attention_mask = torch.ones((2, 10, 10))
	multi_head_attention = MultiHeadAttention(1024, 128, 8)
	out, _ = multi_head_attention(hidden_state, attention_mask)
	print(f"Output shape: {out.shape}")

	# Test KV Cache
	print("\nTesting KV Cache:")
	# 1. Prefill
	seq_len_1 = 5
	input_1 = hidden_state[:, :seq_len_1, :]
	mask_1 = attention_mask[:, :seq_len_1, :seq_len_1]
	out_1, kv_cache = multi_head_attention(input_1, mask_1)
	print(f"Step 1 output shape: {out_1.shape}")
	
	# 2. Decode next token
	input_2 = hidden_state[:, seq_len_1:seq_len_1+1, :]
	mask_2 = torch.ones((2, 1, seq_len_1 + 1)) 
	out_2, kv_cache = multi_head_attention(input_2, mask_2, kv_cache=kv_cache)
	print(f"Step 2 output shape: {out_2.shape}")
	print(f"KV Cache shape: k={kv_cache[0].shape}, v={kv_cache[1].shape}")
