import torch
import torch.nn as nn


class ChamferMatching(nn.Module):

    def __init__(self, args):
        super(ChamferMatching, self).__init__()
        self.args = args

    def forward(self, similarity):
        """ Vote by averaging the diagonal values.
        :input: the similarity matrix as a tensor of size
        (way * query_per_class, way * shot, query_seq_len, query_seq_len)
        """
        max_similarity, _ = torch.max(similarity, dim=-1)
        return torch.mean(max_similarity, (-1))  # (way * query_per_class, way * shot)
