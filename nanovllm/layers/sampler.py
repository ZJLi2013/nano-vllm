import torch
from torch import nn


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))  # 温度缩放
        probs = torch.softmax(logits, dim=-1)  # softmax 概率
        sample_tokens = probs.div_(
            torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
        ).argmax(
            dim=-1
        )  # Gumbel-max 采样
        return sample_tokens


"""
Gumbel-max 采样:
    1. 每个采样点，初始化 exp(1) 作为 noise
    2. probs/noise ~= log(probs) - log(noise) 
    3. 取 argmax 返回
"""
