import torch
import torch.nn as nn


class DiagonalMatching(nn.Module):

    def __init__(self, args):
        super(DiagonalMatching, self).__init__()
        self.args = args

    def forward(self, similarity):
        """ Vote by averaging the diagonal values.
        :input: the similarity matrix as a tensor of size
        (way * query_per_class, way * shot, query_seq_len, query_seq_len)
        """
        diag_frame_sim = torch.diagonal(similarity, offset=0, dim1=-1, dim2=-2)
        return torch.mean(diag_frame_sim, (-1))  # (way * query_per_class, way * shot)
