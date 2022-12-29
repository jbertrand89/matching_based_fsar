import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from itertools import combinations

from utils import extract_class_indices, cos_sim
from .few_shot_head import CNN_FSHead


class PositionalEncoding(nn.Module):
    """
    Positional encoding from the Transformer paper.
    """

    def __init__(self, d_model, dropout, max_len=5000, pe_scale_factor=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe_scale_factor = pe_scale_factor
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) * self.pe_scale_factor
        pe[:, 1::2] = torch.cos(position * div_term) * self.pe_scale_factor
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class TemporalCrossTransformer(nn.Module):
    """
    A temporal cross transformer for a single tuple cardinality. E.g. pairs or triples.
    """

    def __init__(self, args, temporal_set_size=3):
        super(TemporalCrossTransformer, self).__init__()

        self.args = args
        self.temporal_set_size = temporal_set_size

        max_len = int(self.args.seq_len * 1.5)
        self.pe = PositionalEncoding(self.args.trans_linear_in_dim, self.args.trans_dropout,
                                     max_len=max_len)

        self.k_linear = nn.Linear(self.args.trans_linear_in_dim * temporal_set_size,
                                  self.args.trans_linear_out_dim)  # .cuda()
        self.v_linear = nn.Linear(self.args.trans_linear_in_dim * temporal_set_size,
                                  self.args.trans_linear_out_dim)  # .cuda()

        self.norm_k = nn.LayerNorm(self.args.trans_linear_out_dim)
        self.norm_v = nn.LayerNorm(self.args.trans_linear_out_dim)

        self.class_softmax = torch.nn.Softmax(dim=1)

        # generate all tuples
        frame_idxs = [i for i in range(self.args.seq_len)]
        frame_combinations = combinations(frame_idxs, temporal_set_size)
        self.tuples = nn.ParameterList(
            [nn.Parameter(torch.tensor(comb), requires_grad=False) for comb in frame_combinations])
        self.tuples_len = len(self.tuples)

        self.global_temperature = torch.nn.Parameter(
            torch.tensor(float(self.args.voting_temperature)),
            requires_grad=not self.args.voting_global_temperature_fixed)
        self.temperature_weight = torch.nn.Parameter(
            float(self.args.voting_global_weights_const_value) * torch.ones(1),
            requires_grad=not self.args.voting_global_weights_fixed)
        print(f"self.global_temperature {self.global_temperature}")
        print(f"self.temperature_weight {self.temperature_weight}")

    def forward(self, support_set, support_labels, queries):
        n_queries = queries.shape[0]
        n_support = support_set.shape[0]

        # static pe
        support_set = self.pe(support_set)
        queries = self.pe(queries)

        # construct new queries and support set made of tuples of images after pe
        s = [torch.index_select(support_set, -2, p).reshape(n_support, -1) for p in self.tuples]
        q = [torch.index_select(queries, -2, p).reshape(n_queries, -1) for p in self.tuples]
        support_set = torch.stack(s, dim=-2)
        queries = torch.stack(q, dim=-2)

        # apply linear maps
        support_set_ks = self.k_linear(support_set)
        queries_ks = self.k_linear(queries)
        support_set_vs = self.v_linear(support_set)
        queries_vs = self.v_linear(queries)

        # apply norms where necessary
        mh_support_set_ks = self.norm_k(support_set_ks)
        mh_queries_ks = self.norm_k(queries_ks)
        mh_support_set_vs = support_set_vs
        mh_queries_vs = queries_vs

        unique_labels = torch.unique(support_labels)

        # init tensor to hold distances between every support tuple and every target tuple
        all_distances_tensor = torch.zeros(n_queries, self.args.way, device=queries.device)

        for label_idx, c in enumerate(unique_labels):
            # select keys and values for just this class
            class_k = torch.index_select(mh_support_set_ks, 0,
                                         extract_class_indices(support_labels, c))
            class_v = torch.index_select(mh_support_set_vs, 0,
                                         extract_class_indices(support_labels, c))
            k_bs = class_k.shape[0]

            class_scores = torch.matmul(mh_queries_ks.unsqueeze(1),
                                        class_k.transpose(-2, -1)) / math.sqrt(
                self.args.trans_linear_out_dim)

            # reshape etc. to apply a softmax for each query tuple
            class_scores = class_scores.permute(0, 2, 1, 3)
            class_scores = class_scores.reshape(n_queries, self.tuples_len, -1)
            class_scores = [self.class_softmax(class_scores[i]) for i in range(n_queries)]
            class_scores = torch.cat(class_scores)
            class_scores = class_scores.reshape(n_queries, self.tuples_len, -1, self.tuples_len)
            class_scores = class_scores.permute(0, 2, 1, 3)

            # get query specific class prototype
            query_prototype = torch.matmul(class_scores, class_v)
            query_prototype = torch.sum(query_prototype, dim=1)

            # calculate distances from queries to query-specific class prototypes
            diff = mh_queries_vs - query_prototype
            norm_sq = torch.norm(diff, dim=[-2, -1]) ** 2
            distance = torch.div(norm_sq, self.tuples_len)

            # multiply by -1 to get logits
            distance = distance * -1
            c_idx = c.long()
            distance *= self.global_temperature
            distance *= self.temperature_weight
            all_distances_tensor[:, c_idx] = distance

        return_dict = {'logits': all_distances_tensor}

        return return_dict


class CNN_TRX(CNN_FSHead):
    """
    Backbone connected to Temporal Cross Transformers of multiple cardinalities.
    """

    def __init__(self, args):
        super(CNN_TRX, self).__init__(args)

        # fill default args
        self.args.trans_linear_out_dim = 1152
        self.args.trans_dropout = 0.1

        self.transformers = nn.ModuleList(
            [TemporalCrossTransformer(args, s) for s in args.temp_set])

    def forward(self, support_images, support_labels, target_images):
        # support_features, target_features = self.get_feats(support_images, target_images)
        support_features, target_features = support_images, target_images
        all_logits = [t(support_features, support_labels, target_features)['logits'] for t in
                      self.transformers]
        all_logits = torch.stack(all_logits, dim=-1)
        sample_logits = all_logits
        sample_logits = torch.mean(sample_logits, dim=[-1])

        return_dict = {'logits': sample_logits}
        return return_dict

    def distribute_model(self):
        """
        Distributes the CNNs over multiple GPUs. Leaves TRX on GPU 0.
        :return: Nothing
        """
        CNN_FSHead.distribute_model()
        self.transformers.cuda(0)
