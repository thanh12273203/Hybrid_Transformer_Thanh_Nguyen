from typing import List

from torch import nn, Tensor


class InteractionEmbedding(nn.Module):
    def __init__(
        self,
        num_interaction_features: int = 4,
        pair_embed_dims: List[int] = [64, 64, 64, 8]
    ):
        super(InteractionEmbedding, self).__init__()
        input_dim = num_interaction_features
        layers = [nn.BatchNorm1d(input_dim)]
        for dim in pair_embed_dims:
            layers.extend([
                nn.Conv1d(input_dim, dim, kernel_size=1),
                nn.BatchNorm1d(dim),
                nn.GELU()
            ])
            input_dim = dim
        self.embed = nn.Sequential(*layers)

    def forward(self, U: Tensor) -> Tensor:
        B, N, _, F = U.shape  # (batch_size, max_num_particles, max_num_particles, num_interaction_features)
        U = U.view(B, N * N, F).transpose(1, 2)  # (B, F, N * N)
        U = self.embed(U)  # (B, num_heads, N * N)
        U = U.view(B * U.shape[1], N, N)  # (B * num_heads, N, N)

        return U