import torch
import torch.nn as nn


class PointWiseFeedForward(nn.Module):
    def __init__(self, dim_embed: int, dff: int) -> None:
        super().__init__()
        self.fully_connected_1 = nn.Linear(dim_embed, dff)
        self.relu = nn.ReLU()
        self.fully_connected_2 = nn.Linear(dff, dim_embed)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fully_connected_1(x)
        x = self.relu(x)
        x = self.fully_connected_2(x)
        return x