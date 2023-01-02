import torch
import torch.nn as nn


class MaxMatching(nn.Module):
    """
    Module for the max matching function. It selects the maximum frame-to-frame similarity value.
    """
    def __init__(self, args):
        super(MaxMatching, self).__init__()
        self.args = args

    def forward(self, similarity):
        """ Forward pass

        :param similarity: the frame to frame similarity matrix, it is a tensor of size
          [query count, support count, query clip count, support clip count]
        :return: the video to video similarity score, it is a tensor of size
          [query count, support count]
        """
        max_similarity, _ = torch.max(similarity, dim=-1)
        x, _ = torch.max(max_similarity, dim=-1)
        return x
