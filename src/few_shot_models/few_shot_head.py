import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from einops import rearrange
from itertools import combinations


class FewShotHead(nn.Module):
    """
    Base class which handles a few-shot method.

    This class was first introduced in https://github.com/tobyperrett/few-shot-action-recognition
    and adapted for r2+1d precomputed features. In this version, it is fed with features as input
    (through a video loader). But it can also be used with a clips or images as inputs, and it
    requires to adapt the method get_backbone.
    """
    def __init__(self, args):
        super(FewShotHead, self).__init__()
        self.train()
        self.args = args

        # initialize the backbone if we don't load the features
        if not self.args.load_features:
            self.get_backbone()
        else:
            self.backbone = None

        # defines the clip tuples
        if self.args.clip_tuple_length > 1:
            frame_idxs = list(range(self.args.seq_len))
            frame_combinations = combinations(frame_idxs, self.args.clip_tuple_length)
            tuples = [nn.Parameter(torch.tensor(comb), requires_grad=False) for comb in
                      frame_combinations]
            self.clip_tuples = nn.ParameterList(tuples)
        else:
            self.clip_tuples = None

    def get_backbone(self):
        """ Initializes the backbone for different resnet architectures and preload the weights
        """
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
        """Takes in images from the support set and query video and returns CNN features.
        If the load_feature is true, the video loader already returns the features and there is no
        need for extra computation

        :param support_images: the support images if self.args.load_features is false / the support
          features if self.args.load_features is true
        :param target_images: the query images if self.args.load_features is false / the query
          features if self.args.load_features is true
        :return: the support features
        :return: the query features
        """
        if self.args.load_features:
            return support_images, target_images

        support_features = self.backbone(support_images).squeeze()
        target_features = self.backbone(target_images).squeeze()

        dim = int(support_features.shape[1])

        support_features = support_features.reshape(-1, self.args.seq_len, dim)
        target_features = target_features.reshape(-1, self.args.seq_len, dim)

        return support_features, target_features

    def get_video_to_class_similarity(self, video_to_video_similarity):
        """Aggregates the video-to-video scores into video-to-class when there are more than one
        example per class. By default, the mean aggregation is used.

        :param video_to_video_similarity: video-to-video similarity scores
        :return: the video-to-class scores
        """
        results_multi_shot_ordered_per_class = rearrange(
            video_to_video_similarity, "a (b c) -> a b c", b=self.args.way)
        # (way * query_per_class, way, shot)

        return torch.mean(results_multi_shot_ordered_per_class, dim=2)
        # (way * query_per_class, support way)

    def forward(self, support_images, support_labels, target_images):
        """
        Should return a dict containing logits which are required for computing accuracy. Dict can
        also contain other info needed to compute the loss. E.g. inter class distances.

        :param support_images: the support clips if self.args.load_features is false / the support
          features if self.args.load_features is true
        :param support_labels: the labels of the support images
        :param target_images: the query clips if self.args.load_features is false / the query
          features if self.args.load_features is true
        :return: the logits
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
        Computes the loss between the logits and the true labels. By default, it is the cross
        entropy loss.

        :param task_dict: dictionary containing the ground truth labels stored at key "target_labels
        :param model_dict: dictionary containing the logits stored at key logits
        :return: the cross entropy loss
        """
        return F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long())
