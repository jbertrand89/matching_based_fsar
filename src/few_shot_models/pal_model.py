import torch
import torch.nn.functional as F
from torch.nn.modules.activation import MultiheadAttention
from einops import rearrange

from utils import extract_class_indices, cos_sim
from .few_shot_head import CNN_FSHead


class CNN_PAL(CNN_FSHead):
    """
    PAL with a CNN backbone. Cosine similarity as distance measure.
    """

    def __init__(self, args):
        super(CNN_PAL, self).__init__(args)
        self.mha = MultiheadAttention(embed_dim=self.args.trans_linear_in_dim, num_heads=1,
                                      dropout=0)
        self.cos_sim = torch.nn.CosineSimilarity()
        self.loss_lambda = 1

    def forward(self, support_images, support_labels, target_images):
        support_features, target_features = self.get_feats(support_images, target_images)
        unique_labels = torch.unique(support_labels)

        support_features = torch.mean(support_features, dim=1)
        target_features = torch.mean(target_features, dim=1)

        support_features = rearrange(support_features, 'n d -> n 1 d')
        target_features = rearrange(target_features, 'n d -> n 1 d')

        support_features = support_features + \
                           self.mha(support_features, support_features, support_features)[0]
        target_features = target_features + \
                          self.mha(target_features, support_features, support_features)[0]

        support_features = rearrange(support_features, 'b 1 d -> b d')
        target_features = rearrange(target_features, 'b 1 d -> b d')

        prototypes = [torch.mean(
            torch.index_select(support_features, 0, extract_class_indices(support_labels, c)),
            dim=0) for c in unique_labels]
        prototypes = torch.stack(prototypes)

        q_s_sim = cos_sim(target_features, prototypes)

        return_dict = {'logits': q_s_sim}

        return return_dict

    def loss(self, task_dict, model_dict):
        """
        Computes cross entropy loss on the logits, and the additional loss between the queries and their correct classes.
        """
        q_s_sim = model_dict["logits"]
        l_meta = F.cross_entropy(q_s_sim, task_dict["target_labels"].long())

        pcc_q_s_sim = q_s_sim
        pcc_q_s_sim = torch.sigmoid(q_s_sim)

        unique_labels = torch.unique(task_dict["support_labels"])
        total_q_c_sim = torch.sum(pcc_q_s_sim, dim=0) + 0.1

        q_c_sim = [torch.sum(torch.index_select(pcc_q_s_sim, 0,
                                                extract_class_indices(task_dict["target_labels"],
                                                                      c)), dim=0) for c in
                   unique_labels]
        q_c_sim = torch.stack(q_c_sim)
        q_c_sim = torch.diagonal(q_c_sim)
        q_c_sim = torch.div(q_c_sim, total_q_c_sim)

        l_pcc = - torch.mean(torch.log(q_c_sim))

        return l_meta + self.loss_lambda * l_pcc

