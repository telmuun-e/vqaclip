import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, max_len: int, dim_embed: int, n_head: int) -> None:
        super().__init__()
        self.max_len = max_len
        self.dim_embed = dim_embed
        self.n_head = n_head

        self.q = nn.Linear(dim_embed, dim_embed)
        self.k = nn.Linear(dim_embed, dim_embed)
        self.v = nn.Linear(dim_embed, dim_embed)

        self.soft_max = nn.Softmax(dim=-1)
        self.fc = nn.Linear(dim_embed, dim_embed)


    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, x_mask: torch.Tensor = None) -> torch.Tensor:
        b_size = q.size(0)
        assert self.dim_embed % self.n_head == 0
        d_v = int(self.dim_embed / self.n_head)

        q_x = self.q(q).view(b_size, self.max_len, self.n_head, d_v).transpose(1, 2)
        k_x = self.k(k).view(b_size, self.max_len, self.n_head, d_v).transpose(1, 2)
        v_x = self.v(v).view(b_size, self.max_len, self.n_head, d_v).transpose(1, 2)

        q_k = torch.matmul(q_x, k_x.transpose(2, 3))
        q_k *= (self.dim_embed**0.5)

        if x_mask is not None:
            x_mask = x_mask.unsqueeze(1)
            q_k = q_k.masked_fill(x_mask==False, 1e-9)
        soft = self.soft_max(q_k)
        out = torch.matmul(soft, v_x)
        
        out = out.transpose(1, 2)
        out = out.contiguous().view(b_size, self.max_len, -1)
        out = self.fc(out)  
        return out