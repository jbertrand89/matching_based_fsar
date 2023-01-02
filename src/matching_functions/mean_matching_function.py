import torch
import torch.nn as nn


class MeanMatching(nn.Module):
    def __init__(self, args):
        super(MeanMatching, self).__init__()
        self.args = args

    def forward(self, similarity):
        return torch.mean(torch.sum(similarity, dim=-1), dim=-1)
