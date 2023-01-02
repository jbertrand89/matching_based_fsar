from einops import rearrange
import torch.nn as nn


class LinearMatching(nn.Module):
    """ Module for the linear matching function. It flattens the frame-to-frame similarity matrix
    and apply an linear layer on top of it.
    """
    def __init__(self, args):
        super(LinearMatching, self).__init__()
        self.args = args
        self.linear = nn.Linear(self.args.seq_len * self.args.seq_len, 1, bias=False)

    def forward(self, similarity):
        """ Forward pass

        :param similarity: the frame to frame similarity matrix, it is a tensor of size
          [query count, support count, query clip count, support clip count]
        :return: the video to video similarity score, it is a tensor of size
          [query count, support count]
        """
        similarity_flattened = rearrange(similarity, "q s lq ls -> (q s) (lq ls)")
        x = self.linear(similarity_flattened)
        x = rearrange(x, "(q s) 1 -> q s", q=similarity.shape[0], s=similarity.shape[1])
        return x
