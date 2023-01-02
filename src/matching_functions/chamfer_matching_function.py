import torch
import torch.nn as nn


class ChamferMatching(nn.Module):
    """
    Module for the chamfer matching function, in its non symmetric variant.

    Note: To apply the symmetric variant, first transpose the frame-to-frame similarity matrix and
    apply chamfer matching on it and sum it with the non symmetric variant. When applied after a FCN
    like in ViSiL, the filtering FCN must be applied after the transpose and before the matching.
    It is not equivalent to apply the transpose before and after the filtering.
    """
    def __init__(self, args):
        super(ChamferMatching, self).__init__()
        self.args = args

    def forward(self, similarity):
        """ Forward pass

        :param similarity: the frame to frame similarity matrix, it is a tensor of size
          [query count, support count, query clip count, support clip count]
        :return: the video to video similarity score, it is a tensor of size
          [query count, support count]
        """
        max_similarity, _ = torch.max(similarity, dim=-1)
        return torch.mean(max_similarity, (-1))
