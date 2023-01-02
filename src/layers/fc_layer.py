import torch.nn as nn


class FCLayer(nn.Module):
    """
      Multilayer Perceptron.
    """
    def __init__(self, input_d, output_d=1, with_batch_norm=False, with_sigmoid=False):
        super(FCLayer, self).__init__()

        self.with_batch_norm = with_batch_norm
        self.with_sigmoid = with_sigmoid

        self.batch_norm = nn.BatchNorm1d(num_features=input_d)
        self.linear = nn.Linear(input_d, output_d, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, similarity):
        """Forward pass"""
        x = similarity
        if self.with_batch_norm:
            x = self.batch_norm(x)

        x = self.linear(x)

        if self.with_sigmoid:
            x = self.sigmoid(x).clone()

        return x

    def __repr__(self):
        s = ""
        if self.with_batch_norm:
            s += f"(batch norm) {self.batch_norm}"

        s += f"(linear) {self.linear} weights {self.linear.weight}"
        # print(f"(linear) {self.linear} weights {self.linear.weight} bias {self.linear.bias}")

        if self.with_sigmoid:
            s += f"(sigmoid) {self.sigmoid}"
        return s

