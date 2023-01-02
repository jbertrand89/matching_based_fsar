import torch
import torch.nn as nn


class MaxMatching(nn.Module):

    def __init__(self, args):
        super(MaxMatching, self).__init__()
        self.args = args

    def forward(self, similarity):
        """ Vote by averaging the diagonal values.
        :input: the similarity matrix as a tensor of size
        (way * query_per_class, way * shot, query_seq_len, query_seq_len)
        """
        max_similarity, _ = torch.max(similarity, dim=-1)
        x, _ = torch.max(max_similarity, dim=-1)
        return x
