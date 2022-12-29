import torch
import torch.nn.functional as F
from utils import extract_class_indices, cos_sim
from einops import rearrange

from .few_shot_head import CNN_FSHead


def OTAM_cum_dist(dists, lbda=0.1):
    """
    Calculates the OTAM distances for sequences in one direction (e.g. query to support).
    :input: Tensor with frame similarity scores of shape [n_queries, n_support, query_seq_len, support_seq_len]
    TODO: clearn up if possible - currently messy to work with pt1.8. Possibly due to stack operation?
    """
    dists = F.pad(dists, (1, 1), 'constant', 0)

    cum_dists = torch.zeros(dists.shape, device=dists.device)

    # top row
    for m in range(1, dists.shape[3]):
        # cum_dists[:,:,0,m] = dists[:,:,0,m] - lbda * torch.log( torch.exp(- cum_dists[:,:,0,m-1]))
        # paper does continuous relaxation of the cum_dists entry, but it trains faster without, so using the simpler version for now:
        cum_dists[:, :, 0, m] = dists[:, :, 0, m] + cum_dists[:, :, 0, m - 1]

        # remaining rows
    for l in range(1, dists.shape[2]):
        # first non-zero column
        cum_dists[:, :, l, 1] = dists[:, :, l, 1] - lbda * torch.log(
            torch.exp(- cum_dists[:, :, l - 1, 0] / lbda) + torch.exp(
                - cum_dists[:, :, l - 1, 1] / lbda) + torch.exp(- cum_dists[:, :, l, 0] / lbda))

        # middle columns
        for m in range(2, dists.shape[3] - 1):
            cum_dists[:, :, l, m] = dists[:, :, l, m] - lbda * torch.log(
                torch.exp(- cum_dists[:, :, l - 1, m - 1] / lbda) + torch.exp(
                    - cum_dists[:, :, l, m - 1] / lbda))

        # last column
        # cum_dists[:,:,l,-1] = dists[:,:,l,-1] - lbda * torch.log( torch.exp(- cum_dists[:,:,l-1,-2] / lbda) + torch.exp(- cum_dists[:,:,l,-2] / lbda) )
        cum_dists[:, :, l, -1] = dists[:, :, l, -1] - lbda * torch.log(
            torch.exp(- cum_dists[:, :, l - 1, -2] / lbda) + torch.exp(
                - cum_dists[:, :, l - 1, -1] / lbda) + torch.exp(- cum_dists[:, :, l, -2] / lbda))

    return cum_dists[:, :, -1, -1]


class CNN_OTAM(CNN_FSHead):
    """
    OTAM with a CNN backbone.
    """

    def __init__(self, args):
        super(CNN_OTAM, self).__init__(args)
        if self.args.backbone in {"r2+1d_fc"}:
            self.fc = torch.nn.Linear(args.trans_linear_in_dim, args.fc_dimension)
            self.layer_norm = torch.nn.LayerNorm(self.args.fc_dimension)

        self.global_temperature = torch.nn.Parameter(
            torch.tensor(float(self.args.voting_temperature)),
            requires_grad=not self.args.voting_global_temperature_fixed)

        self.temperature_weight = torch.nn.Parameter(
            float(self.args.voting_global_weights_const_value) * torch.ones(1),
            requires_grad=not self.args.voting_global_weights_fixed)
        print(f"self.temperature_weight {self.temperature_weight}")

    def forward(self, support_images, support_labels, target_images):
        # support_features, target_features = self.get_feats(support_images, target_images)
        support_features, target_features = support_images, target_images

        unique_labels = torch.unique(support_labels)
        n_queries = target_features.shape[0]
        n_support = support_features.shape[0]

        if self.args.backbone in {"r2+1d_fc"}:
            support_features = self.fc(support_features)
            target_features = self.fc(target_features)
            support_features = self.layer_norm(support_features)
            target_features = self.layer_norm(target_features)

        support_features = rearrange(support_features, 'b s d -> (b s) d')
        target_features = rearrange(target_features, 'b s d -> (b s) d')

        frame_sim = cos_sim(target_features, support_features)
        frame_dists = 1 - frame_sim

        dists = rearrange(frame_dists, '(tb ts) (sb ss) -> tb sb ts ss', tb=n_queries, sb=n_support)

        # calculate query -> support and support -> query
        cum_dists = OTAM_cum_dist(dists) + OTAM_cum_dist(
            rearrange(dists, 'tb sb ts ss -> tb sb ss ts'))

        class_dists = [
            torch.mean(torch.index_select(cum_dists, 1, extract_class_indices(support_labels, c)),
                       dim=1) for c in unique_labels]
        class_dists = torch.stack(class_dists)
        class_dists = rearrange(class_dists, 'c q -> q c')
        class_dists *= self.temperature_weight  # learnt weight
        class_dists *= self.global_temperature  # fixed temperature
        return_dict = {'logits': - class_dists}
        return return_dict

    def loss(self, task_dict, model_dict):
        return F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long())

