import torch
import torch.nn as nn

from vqaclip.model.attention import MultiHeadAttention
from vqaclip.model.feed_forward import PointWiseFeedForward


class TransformerBlock(nn.Module):
    def __init__(self, max_len: int, dim_embed: int, n_head: int, dff: int, rate: float =0.1) -> None:
        super().__init__()
        self.multi_attention = MultiHeadAttention(max_len, dim_embed, n_head)
        self.fead_forward = PointWiseFeedForward(dim_embed, dff)

        self.layer_norm_1 = nn.LayerNorm(dim_embed, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(dim_embed, eps=1e-6)

        self.dropout_1 = nn.Dropout(rate)
        self.dropout_2 = nn.Dropout(rate)


    def forward(self, x: torch.Tensor, x_mask: torch.Tensor = None) -> torch.Tensor:
        output = self.multi_attention(x, x, x, x_mask)
        output = self.dropout_1(output)
        output += x
        output = self.layer_norm_1(output)
        residual = output
        
        output = self.fead_forward(output)
        output = self.dropout_2(output)
        output += residual
        output = self.layer_norm_2(output)
        return output