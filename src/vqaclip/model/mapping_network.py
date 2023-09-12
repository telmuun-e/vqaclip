import torch
import torch.nn as nn

from vqaclip.model.transformer import Transformer


class MappingNetwork(nn.Module):
    def __init__(self, max_len: int, dim_clip: int, dim_embedding: int, prefix_length: int, clip_length: int, dff: int, num_layers: int = 8) -> None:
        super().__init__()
        self.clip_length = clip_length
        self.transformer = Transformer(max_len, dim_embedding, 8, dff, num_layers)
        self.linear = nn.Linear(dim_clip, clip_length * dim_embedding)
        self.prefix_const = nn.Parameter(torch.randn(prefix_length, dim_embedding), requires_grad=True)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x).view(x.shape[0], self.clip_length, -1)
        prefix = self.prefix_const.unsqueeze(0).expand(x.shape[0], *self.prefix_const.shape)
        prefix = torch.cat((x, prefix), dim=1)
        out = self.transformer(prefix)[:, self.clip_length:]
        return out