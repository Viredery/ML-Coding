import torch
import torch.nn as nn
import torch.nn.functional as F



class Network(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        output_dim: int,
        min_period: float = 4e-3,
        max_period: float = 4.0,
    ):
        super().__init__()

        assert input_dim % 2 == 0
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.min_period = min_period
        self.max_period = max_period

        self.linear = nn.Linear(self.input_dim * 2, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, input_dim]
            t: [batch_size]. The range of t in (0.0, 1.0)
        Returns:
            output: [batch_size, output_dim]
        """
        t_emb = self.create_sinusoidal_embedding(t)
        x = torch.cat([x, t_emb], dim=-1)
        x = self.linear(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x

    def create_sinusoidal_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [batch_size]. The range of t in (0.0, 1.0)
        Returns:
            time_emb: [batch_size, input_dim]
        """
        T = self.input_dim // 2
        t_scale = torch.arange(T, dtype=torch.float32, device=t.device) / (T - 1)

        periods = self.min_period * (self.max_period / self.min_period) ** t_scale
        angles = 2 * torch.pi / periods

        sin_value = torch.sin(angles.unsqueeze(0) * t.unsqueeze(-1).to(torch.float32))
        cos_value = torch.cos(angles.unsqueeze(0) * t.unsqueeze(-1).to(torch.float32))
        time_emb = torch.cat([sin_value, cos_value], dim=-1)

        return time_emb


class LinearScheduler(nn.Module):
    def __init__(self, T: int):
        super().__init__()

        self.T = T
        alpha = torch.sqrt(1 - 0.02 * torch.arange(1, T + 1) / T)
        beta = torch.sqrt(1 - alpha ** 2)
        alpha_bar = torch.cumprod(alpha, dim=0)
        beta_bar = torch.sqrt(1 - alpha_bar ** 2)

        t = torch.linspace(0.0, 1.0, T, dtype=torch.float32)

        self.register_buffer('alpha', alpha)
        self.register_buffer('beta', beta)
        self.register_buffer('alpha_bar', alpha_bar)
        self.register_buffer('beta_bar', beta_bar)
        self.register_buffer('t', t)


class DDPM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        T: int,
    ):
        super().__init__()

        assert input_dim == output_dim
        assert T > 0

        self.input_dim = input_dim
        self.network = Network(input_dim, hidden_dim, output_dim)
        self.linear_scheduler = LinearScheduler(T)

    def sample_noise(self, x: torch.Tensor):
        noise = torch.normal(
            mean=0.0, 
            std=1.0,
            size=x.shape,
            dtype=torch.float32,
            device=x.device,
        )
        return noise

    def sample_time(self, x: torch.Tensor):
        return torch.randint(0, self.linear_scheduler.T, (x.shape[0],), device=x.device, dtype=torch.long)

    def forward_process(self, x: torch.Tensor) -> torch.Tensor:
        t = self.sample_time(x)
        t_cont = self.linear_scheduler.t[t]
        noise = self.sample_noise(x)
        alpha_bar = self.linear_scheduler.alpha_bar.index_select(dim=0, index=t).unsqueeze(-1)
        beta_bar = self.linear_scheduler.beta_bar.index_select(dim=0, index=t).unsqueeze(-1)
        return alpha_bar * x + beta_bar * noise, t_cont, noise

    def reverse_process(self, x: torch.Tensor) -> torch.Tensor:
        for t in range(self.linear_scheduler.T - 1, -1, -1):
            alpha = self.linear_scheduler.alpha[t]
            beta = self.linear_scheduler.beta[t]
            beta_bar = self.linear_scheduler.beta_bar[t]

            t_cont = self.linear_scheduler.t[t].expand(x.shape[0]).to(x.device)
            pred_noise = self.network(x, t_cont)

            x = (x - (beta ** 2 / beta_bar) * pred_noise) / alpha

            if t > 0:
                beta_bar_prev = self.linear_scheduler.beta_bar[t - 1]
                sigma = beta * (beta_bar_prev / beta_bar)
                noise = self.sample_noise(x)
                x = x + sigma * noise

        return x

    def train_step(self, x: torch.Tensor) -> torch.Tensor:
        forward_process, t, noise = self.forward_process(x)
        pred_noise = self.network(forward_process, t)
        loss = F.mse_loss(pred_noise, noise)
        return loss

if __name__ == '__main__':
    # Test forward and reverse process
    input_dim = 128
    hidden_dim = 256
    output_dim = 128
    T = 1000

    ddpm = DDPM(input_dim, hidden_dim, output_dim, T)
    x = torch.randn(1, input_dim)
    forward_process, t, noise = ddpm.forward_process(x)
    reverse_process = ddpm.reverse_process(forward_process)
    print(forward_process.shape)
    print(reverse_process.shape)
