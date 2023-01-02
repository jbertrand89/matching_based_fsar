import torch
import torch.nn as nn
from einops import rearrange


class VisilFCN(nn.Module):

    def __init__(self, args):
        super(VisilFCN, self).__init__()
        self.args = args
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.pool1 = torch.nn.MaxPool2d((2, 2), stride=2, padding=0, dilation=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.pool2 = torch.nn.MaxPool2d((2, 2), stride=2, padding=0, dilation=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.conv4 = nn.Conv2d(128, 1, kernel_size=(1, 1), stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.hardtanh = nn.Hardtanh()

    def forward(self, similarity):
        """ Vote by averaging the diagonal values.
        :input: the similarity matrix as a tensor of size
        (way * query_per_class, way * shot, query_seq_len, query_seq_len)
        """
        x = rearrange(similarity, 'q s lq ls -> (q s) 1 lq ls')
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.relu(x)
        x = self.conv3(x)
        output_similarity = self.conv4(x)
        output_similarity = self.hardtanh(output_similarity)
        output_similarity = rearrange(
            output_similarity, '(q s) 1 lq ls -> q s lq ls', q=similarity.shape[0])
        return output_similarity

