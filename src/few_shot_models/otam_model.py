import torch
import torch.nn.functional as F
from utils import extract_class_indices, cos_sim
from einops import rearrange
from src.matching_functions.matching_function_utils import get_matching_function
from src.matching_functions.visil_fcn import VisilFCN

from .few_shot_head import CNN_FSHead


class CNN_OTAM(CNN_FSHead):
    """
    OTAM with a CNN backbone.
    """
    def __init__(self, args):
        super(CNN_OTAM, self).__init__(args)
        if self.args.backbone in {"r2+1d_fc"}:
            self.fc = torch.nn.Linear(
                args.backbone_feature_dimension * self.args.clip_tuple_length,
                args.feature_projection_dimension)
            self.layer_norm = torch.nn.LayerNorm(self.args.feature_projection_dimension)

        self.matching_function = get_matching_function(args=self.args)

        if self.args.visil:
            self.visil_fcn = VisilFCN(self.args)

        self.global_temperature = torch.nn.Parameter(
            torch.tensor(float(self.args.matching_global_temperature)),
            requires_grad=not self.args.matching_global_temperature_fixed)

        self.temperature_weight = torch.nn.Parameter(
            float(self.args.matching_temperature_weight) * torch.ones(1),
            requires_grad=not self.args.matching_temperature_weight_fixed)
        print(f"self.temperature_weight {self.temperature_weight}")

    def get_frame_to_frame_similarity_matrix(self, support_images, target_images):
        support_features, target_features = self.get_feats(support_images, target_images)

        n_queries = target_features.shape[0]
        n_support = support_features.shape[0]

        if self.args.clip_tuple_length > 1:
            # create clip tuples for the support
            support = [torch.index_select(support_features, -2, p).reshape(n_support, -1)
                       for p in self.clip_tuples]
            support_features = torch.stack(support, dim=-2)

            # create clip tuples for the target/query
            target = [torch.index_select(target_features, -2, p).reshape(n_queries, -1)
                      for p in self.clip_tuples]
            target_features = torch.stack(target, dim=-2)

        # projection head
        if self.args.backbone in {"r2+1d_fc"}:
            support_features = self.fc(support_features)
            target_features = self.fc(target_features)
            support_features = self.layer_norm(support_features)
            target_features = self.layer_norm(target_features)

        support_features = rearrange(support_features, 'b s d -> (b s) d')
        # (way * shot * seq_len, embedding_dimension)
        target_features = rearrange(target_features, 'b s d -> (b s) d')
        # (way * query_per_class * seq_len, embedding_dimension)

        # cosine similarity
        frame_to_frame_similarity = cos_sim(target_features, support_features)
        # (way * query_per_class * seq_len, way * shot * seq_len)

        frame_to_frame_similarity = rearrange(
            frame_to_frame_similarity, '(qb ql) (sb sl) -> qb sb ql sl', qb=n_queries, sb=n_support)
        # (way * query_per_class, way * shot, query_seq_len, support_seq_len)
        return frame_to_frame_similarity

    def transpose_frame_to_frame_similarity(self, frame_to_frame_similarity):
        """ Transposes the frame to frame similarity. The transposed matrix is used for the
        chamfer-support and chamfer++ matching functions. Otherwise, it is not computed.

        :param frame_to_frame_similarity: input similarity matrix of shape
          (way * query_per_class, way * shot, query clip count (seq_len), support clip count)
        :return: the transposed similarity matrix of shape
          (way * shot, way * query_per_class, query clip count, support clip count)
        """
        if self.args.matching_function in {"chamfer-support", "chamfer++"}:
            return torch.transpose(frame_to_frame_similarity, dim0=-2, dim1=-1)
        else:
            return None

    def filter_similarity(self, frame_to_frame_similarity, frame_to_frame_similarity_transposed):
        """ Apply the fully connected network defined in ViSiL https://arxiv.org/abs/1908.07410

        :param frame_to_frame_similarity: input similarity matrix of shape
         (way * query_per_class, way * shot, query clip count (seq_len), support clip count)
        :param frame_to_frame_similarity_transposed: the transposed similarity matrix of shape
         (way * shot, way * query_per_class, query clip count, support clip count)
         :return:
        """
        if self.args.visil:
            frame_to_frame_similarity = self.visil_fcn.forward(frame_to_frame_similarity)

            if frame_to_frame_similarity_transposed is not None:
                frame_to_frame_similarity_transposed = self.visil_fcn.forward(
                    frame_to_frame_similarity_transposed)

        return frame_to_frame_similarity, frame_to_frame_similarity_transposed

    def joint_similarity(self, frame_to_frame_similarity, frame_to_frame_similarity_transposed):
        if self.args.video_to_class_matching == "joint":
            frame_to_frame_similarity = rearrange(
                frame_to_frame_similarity, "q (w s) lq ls -> q w lq (s ls)", s=self.args.shot)

            if frame_to_frame_similarity_transposed is not None:
                frame_to_frame_similarity_transposed = rearrange(
                    frame_to_frame_similarity_transposed,
                    "q (w s) lq ls -> q w lq (s ls)", s=self.args.shot)

        return frame_to_frame_similarity, frame_to_frame_similarity_transposed

    def get_video_to_video_similarity(
            self, frame_to_frame_similarity, frame_to_frame_similarity_transposed):
        if self.args.matching_function == "chamfer-support":
            video_to_video_similarity = self.matching_function.forward(
                frame_to_frame_similarity_transposed)
        elif self.args.matching_function == "chamfer++":
            video_to_video_similarity = self.matching_function.forward(frame_to_frame_similarity)
            video_to_video_similarity_transposed = self.matching_function.forward(
                frame_to_frame_similarity_transposed)
            video_to_video_similarity = 0.5 * (
                    video_to_video_similarity + video_to_video_similarity_transposed)
        else:
            video_to_video_similarity = self.matching_function.forward(frame_to_frame_similarity)

        return video_to_video_similarity

    def forward(self, support_images, support_labels, target_images):

        frame_to_frame_similarity = self.get_frame_to_frame_similarity_matrix(
            support_images, target_images)

        frame_to_frame_similarity_transposed = self.transpose_frame_to_frame_similarity(
            frame_to_frame_similarity)

        frame_to_frame_similarity, frame_to_frame_similarity_transposed = self.filter_similarity(
            frame_to_frame_similarity, frame_to_frame_similarity_transposed)

        frame_to_frame_similarity, frame_to_frame_similarity_transposed = self.joint_similarity(
            frame_to_frame_similarity, frame_to_frame_similarity_transposed)

        video_to_video_similarity = self.get_video_to_video_similarity(
            frame_to_frame_similarity, frame_to_frame_similarity_transposed)

        video_to_class_similarity = self.get_video_to_class_similarity(video_to_video_similarity)
        video_to_class_similarity *= self.temperature_weight  # learnt temperature weight
        video_to_class_similarity *= self.global_temperature  # fixed temperature

        return_dict = {'logits': video_to_class_similarity}

        return return_dict

    def loss(self, task_dict, model_dict):
        """
        Computes the loss between the logits and the true labels. By default, it is the cross
        entropy loss.

        :param task_dict: dictionary containing the ground truth labels stored at key "target_labels
        :param model_dict: dictionary containing the logits stored at key logits
        :return: the cross entropy loss
        """
        return F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long())

