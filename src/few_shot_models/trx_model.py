import math
import torch
import torch.nn as nn
from itertools import combinations

from .few_shot_head import FewShotHead
# code from the few-shot-action-recognition repo
from utils import extract_class_indices, cos_sim
from model import PositionalEncoding



class TRX_few_shot_model(FewShotHead):
    """
    TRX model for few-shot action recognition.
    This class was first introduced in https://github.com/tobyperrett/few-shot-action-recognition
    and adapted for r2+1d precomputed features. The code is identical except for the parameter
    self.args.temp_set which is now introduced in run_matching.
    """
    def __init__(self, args):
        super(TRX_few_shot_model, self).__init__(args)

        # fill default args
        self.args.trans_linear_out_dim = 1152
        self.args.trans_dropout = 0.1

        self.transformers = nn.ModuleList(
            [TemporalCrossTransformer(args, s) for s in args.temp_set])

    def forward(self, support_clips, support_labels, target_clips):
        """ TRX process:
        - it computes/loads the support and target/query features
        - it created Temporal Cross Transformers of multiple cardinalities.

        :param support_clips: the support clips if self.args.load_features is false / the support
          backbone embeddings if self.args.load_features is true
        :param support_labels: the labels of the support clips
        :param target_clips: the query clips if self.args.load_features is false / the query
          backbone embeddings if self.args.load_features is true
        :return: a dictionary with the video_to_class_similarity as a logit
        """
        support_features, target_features = self.get_feats(support_clips, target_clips)

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
        FewShotHead.distribute_model()
        self.transformers.cuda(0)


class TemporalCrossTransformer(nn.Module):
    """
    A temporal cross transformer for a single tuple cardinality. E.g. pairs or triples.

    This class was first introduced in https://github.com/tobyperrett/few-shot-action-recognition.
    The code is identical except for the temperature parameters self.global_temperature and
    self.temperature_weight.
    """
    def __init__(self, args, temporal_set_size=3):
        super(TemporalCrossTransformer, self).__init__()

        self.args = args
        self.temporal_set_size = temporal_set_size

        max_len = int(self.args.seq_len * 1.5)
        self.pe = PositionalEncoding(self.args.backbone_feature_dimension, self.args.trans_dropout,
                                     max_len=max_len)

        self.k_linear = nn.Linear(self.args.backbone_feature_dimension * temporal_set_size,
                                  self.args.trans_linear_out_dim)  # .cuda()
        self.v_linear = nn.Linear(self.args.backbone_feature_dimension * temporal_set_size,
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
            torch.tensor(float(self.args.matching_global_temperature)),
            requires_grad=not self.args.matching_global_temperature_fixed)
        self.temperature_weight = torch.nn.Parameter(
            float(self.args.matching_temperature_weight) * torch.ones(1),
            requires_grad=not self.args.matching_temperature_weight_fixed)

    def forward(self, support_set, support_labels, queries):
        """
        - adds positional encoding to the features
        - create the clip feature tuples
        - apply the cross attention projection head (K and V)
        - computes the query prototypes
        - computes the distance between the query prototypes and the query projected in the V space

        :param support_set: the support features
        :param support_labels: the labels of the support clips
        :param queries: the query features
        :return: a dictionary with the query to prototype similarity as a logit
        """
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
