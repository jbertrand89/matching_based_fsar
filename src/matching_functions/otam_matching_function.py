import torch.nn as nn
from einops import rearrange
from model import OTAM_cum_dist


class OTAMMatching(nn.Module):
    """
    Module for the OTAM matching function. It finds a path with the best alignment and sum the
    frame-to-frame similarity values corresponding to this path.
    """
    def __init__(self, args):
        super(OTAMMatching, self).__init__()
        self.args = args

    def forward(self, similarity):
        """ Forward pass

        :param similarity: the frame to frame similarity matrix, it is a tensor of size
          [query count, support count, query clip count, support clip count]
        :return: the video to video similarity score, it is a tensor of size
          [query count, support count]
        """
        distance = 1 - similarity
        votes = OTAM_cum_dist(distance) + OTAM_cum_dist(
            rearrange(distance, 'tb sb ts ss -> tb sb ss ts'))
        votes *= 0.5
        return - votes
