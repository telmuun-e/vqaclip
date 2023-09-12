import torch
import torch.nn as nn

from vqaclip.model.transformer_block import TransformerBlock


class Transformer(nn.Module):
    def __init__(self, max_len: int, dim_embed: int, n_head: int, dff: int, n_layers: int, rate: float=0.1) -> None:
        super().__init__()
        self.dim_embed = dim_embed
        self.max_len = max_len
        self.dropout = nn.Dropout(rate)

        self.transformer_stacks = nn.ModuleList([
            TransformerBlock(max_len, dim_embed, n_head, dff, rate) for _ in range(n_layers)
            ])


    def forward(self, x: torch.Tensor, x_mask: torch.Tensor = None) -> torch.torch:
        for transformer_block in self.transformer_stacks:
            output = transformer_block(x, x_mask)
        return output