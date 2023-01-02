import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from einops import rearrange


class CNN_FSHead(nn.Module):
    """
    Base class which handles a few-shot method. Contains a resnet backbone which computes features.
    """

    def __init__(self, args):
        super(CNN_FSHead, self).__init__()
        self.train()
        self.args = args
        if self.args.backbone.startswith("resnet"):
            self.get_backbone()
        else:
            self.backbone = None

    def get_backbone(self):
        if self.args.backbone == "resnet18":
            backbone = models.resnet18(pretrained=True)
        elif self.args.backbone == "resnet34":
            backbone = models.resnet34(pretrained=True)
        elif self.args.backbone == "resnet50":
            backbone = models.resnet50(pretrained=True)

        if self.args.pretrained_backbone is not None:
            checkpoint = torch.load(self.args.pretrained_backbone)
            backbone.load_state_dict(checkpoint)

        last_layer_idx = -1
        self.backbone = nn.Sequential(*list(backbone.children())[:last_layer_idx])

    def get_feats(self, support_images, target_images):
        """
        Takes in images from the support set and query video and returns CNN features.
        """
        support_features = self.backbone(support_images).squeeze()
        target_features = self.backbone(target_images).squeeze()

        dim = int(support_features.shape[1])

        support_features = support_features.reshape(-1, self.args.seq_len, dim)
        target_features = target_features.reshape(-1, self.args.seq_len, dim)

        return support_features, target_features

    def aggregate_multi_shot_faster(self, results_multi_shot):
        """
        Aggregates per class, when k shot is higher than 1. Mean aggregation by default.
        Since the data is originally shuffled, it first needs to be rearranged based on the support_labels
        BE CAREFUL: it will remove the target from the support if self.args.query_in_support is True
        """
        results_multi_shot_ordered_per_class = rearrange(
            results_multi_shot, "a (b c) -> a b c", b=self.args.way)
        # (way * query_per_class, way, shot)

        return torch.mean(results_multi_shot_ordered_per_class, dim=2)
        # (way * query_per_class, support way)

    def forward(self, support_images, support_labels, target_images):
        """
        Should return a dict containing logits which are required for computing accuracy. Dict can also contain
        other info needed to compute the loss. E.g. inter class distances.
        """
        raise NotImplementedError

    def distribute_model(self):
        """
        Use to split the backbone evenly over all GPUs. Modify if you have other components
        """
        if self.args.num_gpus > 1 and self.backbone:
            self.backbone.cuda(0)
            self.backbone = torch.nn.DataParallel(
                self.backbone, device_ids=[i for i in range(0, self.args.num_gpus)])

    def loss(self, task_dict, model_dict):
        """
        Takes in a the task dict containing labels etc.
        Takes in the model output dict, which contains "logits", as well as any other info needed to compute the loss.
        Default is cross entropy loss.
        """
        return F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long())
