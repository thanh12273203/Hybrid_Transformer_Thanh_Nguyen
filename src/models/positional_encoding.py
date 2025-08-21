import torch
from torch import nn, Tensor


class EtaPhiPositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int = 128):
        super(EtaPhiPositionalEncoding, self).__init__()
        # Small MLP to map (eta, phi) -> embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.freqs = torch.exp(
            torch.arange(0, embed_dim // 2, 2).float() * (-torch.log(torch.tensor(10000.0)) / (embed_dim // 2))
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, N, D), where D = embed_dim
        eta = x[..., 1:2]  # (B, N, D)
        phi = x[..., 2:3]  # (B, N, D)
        pos = torch.cat([eta, phi], dim=-1)  # (B, N, D * 2)
        out = self.mlp(pos)  # (B, N, D * 2)

        # Sinusoidal encoding for phi only
        # Broadcast to (B, N, embed_dim // 2)
        phi_val = phi  # (B, N, 1)
        freqs = self.freqs.to(x.device)[None, None, :]  # (1, 1, D // 4)
        sin_phi = torch.sin(phi_val * freqs)
        cos_phi = torch.cos(phi_val * freqs)
        sin_cos = torch.cat([sin_phi, cos_phi], dim=-1)
        sin_cos = sin_cos.repeat(1, 1, 2)
        out = out + sin_cos[..., :out.shape[-1]]

        return out