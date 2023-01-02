from einops import rearrange
import torch.nn as nn

from src.layers.fc_layer import FCLayer


class LinearMatching(nn.Module):
    """Flattens the similarity matrix and apply an fc layer on top of it.

    TODO Used parameters
    """
    def __init__(self, args):
        super(LinearMatching, self).__init__()
        self.args = args
        self.fc = FCLayer(
            64, # todo don't hardcode it
            with_batch_norm=False, #self.args.use_batch_normalization,
            with_sigmoid=False) #self.args.use_sigmoid)

    def forward(self, similarity):
        similarity_flattened = rearrange(similarity, "q s lq ls -> (q s) (lq ls)")
        x = self.fc(similarity_flattened)
        x = rearrange(x, "(q s) 1 -> q s", q=similarity.shape[0], s=similarity.shape[1])
        return x
