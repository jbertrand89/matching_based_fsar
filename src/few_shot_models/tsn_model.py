import torch
from einops import rearrange

from utils import extract_class_indices, cos_sim
from .few_shot_head import CNN_FSHead


class CNN_TSN(CNN_FSHead):
    """
    TSN with a CNN backbone.
    Either cosine similarity or negative norm squared distance.
    Use mean distance from query to class videos.
    """

    def __init__(self, args):
        super(CNN_TSN, self).__init__(args)
        self.norm_sq_dist = False

    def forward(self, support_images, support_labels, target_images):
        support_features, target_features = self.get_feats(support_images, target_images)
        unique_labels = torch.unique(support_labels)

        support_features = torch.mean(support_features, dim=1)
        target_features = torch.mean(target_features, dim=1)

        if self.norm_sq_dist:
            class_prototypes = [torch.mean(
                torch.index_select(support_features, 0, extract_class_indices(support_labels, c)),
                dim=0) for c in unique_labels]
            class_prototypes = torch.stack(class_prototypes)

            diffs = [target_features - class_prototypes[i] for i in unique_labels]
            diffs = torch.stack(diffs)

            norm_sq = torch.norm(diffs, dim=[-1]) ** 2
            distance = - rearrange(norm_sq, 'c q -> q c')
            return_dict = {'logits': distance}

        else:
            class_sim = cos_sim(target_features, support_features)
            class_sim = [torch.mean(
                torch.index_select(class_sim, 1, extract_class_indices(support_labels, c)), dim=1)
                         for c in unique_labels]
            class_sim = torch.stack(class_sim)
            class_sim = rearrange(class_sim, 'c q -> q c')
            return_dict = {'logits': class_sim}

        return return_dict