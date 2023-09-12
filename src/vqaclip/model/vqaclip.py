import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel

from vqaclip.model.mapping_network import MappingNetwork 


class VQAClip(nn.Module):
    def __init__(self, max_len: int, prefix_length: int, prefix_size: int, clip_length: int, num_layers: int) -> None:
        super().__init__()
        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt.eval()
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        self.mapping_network = MappingNetwork(max_len, prefix_size, self.gpt_embedding_size, prefix_length, clip_length, int(self.gpt_embedding_size * 4), num_layers)

    
    def forward(self, prefix: torch.Tensor, input_tokens: torch.Tensor) -> torch.Tensor:
        input_embeddings = self.gpt.transformer.wte(input_tokens)
        mapped_embeddings = self.mapping_network(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        all_embeddings = torch.cat((mapped_embeddings, input_embeddings), dim=1)
        out = self.gpt(inputs_embeds=all_embeddings)
        return out
