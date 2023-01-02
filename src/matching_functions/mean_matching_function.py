import torch
import torch.nn as nn


class MeanMatching(nn.Module):
    """
    Module for the mean matching function. It averages all the values of the frame-to-frame
    similarity matrix.
    """
    def __init__(self, args):
        super(MeanMatching, self).__init__()
        self.args = args

    def forward(self, similarity):
        """ Forward pass

        :param similarity: the frame to frame similarity matrix, it is a tensor of size
          [query count, support count, query clip count, support clip count]
        :return: the video to video similarity score, it is a tensor of size
          [query count, support count]
        """
        return torch.mean(torch.sum(similarity, dim=-1), dim=-1)
