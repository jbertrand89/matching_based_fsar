import torch
import torch.nn as nn


class DiagonalMatching(nn.Module):
    """
    Module for the diagonal matching function. It averages all the values on the diagonal. This
    function assumes that the frames are temporally aligned between the query and the support
    videos.
    """
    def __init__(self, args):
        super(DiagonalMatching, self).__init__()
        self.args = args

    def forward(self, similarity):
        """ Forward pass

        :param similarity: the frame to frame similarity matrix, it is a tensor of size
          [query count, support count, query clip count, support clip count]
        :return: the video to video similarity score, it is a tensor of size
          [query count, support count]
        """
        diag_frame_sim = torch.diagonal(similarity, offset=0, dim1=-1, dim2=-2)
        return torch.mean(diag_frame_sim, (-1))  # (way * query_per_class, way * shot)
