import torch
from einops import rearrange
from src.matching_functions.matching_function_utils import get_matching_function
from src.matching_functions.visil_fcn import VisilFCN
from .few_shot_head import FewShotHead
# code from the few-shot-action-recognition repo
from utils import extract_class_indices, cos_sim


class MatchingBasedFewShotModel(FewShotHead):
    """ Matching-based models for few-shot action recognition.

    It can be used to compute baselines such as:
    - Mean
    - Max
    - Diagonal
    - Linear
    - Chamfer (and our extension, Chamfer++)

    but also existing papers, such as:
    - OTAM (see paper https://arxiv.org/abs/1906.11415)
    - ViSiL (adapted for few-shot learning) (see paper: https://arxiv.org/abs/1908.07410)

    This class was first introduced in https://github.com/tobyperrett/few-shot-action-recognition
    and adapted for using different matching functions and r2+1d precomputed features.
    """
    def __init__(self, args):
        super(MatchingBasedFewShotModel, self).__init__(args)
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

    def get_clip_to_clip_similarity_matrix(self, support_clips, target_clips):
        """ Compute the clip-to-clip similarity matrix from the support clips and the
        target/query clips.
        It first computes the support and target features by feeding the inputs to:
          - a backbone (optional depending on the parameter self.args.load_features)
          - a projection head (optional depending on the parameter self.args.backbone)
        Then it computes the cosine similarity between the support and target features.
        Instead of using the clip features to compute the similarity matrix, it is also possible to
        use tuples of clip features of length self.args.clip_tuple_length. The matrix will then be
        a clip-tuple to clip-tuple similarity matrix.

        :param support_clips: the support clips if self.args.load_features is false / the support
          backbone embeddings if self.args.load_features is true
        :param target_clips: the query clips if self.args.load_features is false / the query
          backbone embeddings if self.args.load_features is true
        :return: the clip-to-clip similarity matrix
        """
        # get backbone embedding
        support_features, target_features = self.get_feats(support_clips, target_clips)

        n_queries = target_features.shape[0]
        n_support = support_features.shape[0]

        # create clip tuples (done after the backbone embedding for performance
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
        clip_to_clip_similarity = cos_sim(target_features, support_features)
        # (way * query_per_class * seq_len, way * shot * seq_len)

        clip_to_clip_similarity = rearrange(
            clip_to_clip_similarity, '(qb ql) (sb sl) -> qb sb ql sl', qb=n_queries, sb=n_support)
        # (way * query_per_class, way * shot, query_seq_len, support_seq_len)
        return clip_to_clip_similarity

    def transpose_similarity(self, clip_to_clip_similarity):
        """ Transposes the clip-to-clip similarity. The transposed matrix is used for the
        chamfer-support and chamfer++ matching functions. Otherwise, it is not computed.

        :param clip_to_clip_similarity: input similarity matrix of shape
          [query video count, support video count, query clip count, support clip count]
        :return: the transposed similarity matrix of shape
          [query video count, support video count, support clip count, query clip count]
        """
        if self.args.matching_function in {"chamfer-support", "chamfer++"}:
            return torch.transpose(clip_to_clip_similarity, dim0=-2, dim1=-1)
        return None

    def filter_similarity(self, clip_to_clip_similarity, clip_to_clip_similarity_transposed):
        """ Applies the fully connected network defined in ViSiL (https://arxiv.org/abs/1908.07410)

        :param clip_to_clip_similarity: input similarity matrix of shape
         [query video count, support video count, query clip count, support clip count]
        :param clip_to_clip_similarity_transposed: the transposed similarity matrix of shape
         [query video count, support video count, support clip count, query clip count]
        :return: clip_to_clip_similarity after the ViSiL layers of shape
         [query video count, support video count, query clip count / 4, support clip count / 4]
        :return: clip_to_clip_similarity_transposed after the ViSiL layers of shape
         [query video count, support video count, support clip count / 4, query clip count / 4]
        """
        if self.args.visil:
            clip_to_clip_similarity = self.visil_fcn(clip_to_clip_similarity)

            if clip_to_clip_similarity_transposed is not None:
                clip_to_clip_similarity_transposed = self.visil_fcn(
                    clip_to_clip_similarity_transposed)

        return clip_to_clip_similarity, clip_to_clip_similarity_transposed

    def concatenate_support_examples(self, clip_to_clip_similarity):
        """ Concatenates the temporal similarity matrices between the query and all support examples

        :param clip_to_clip_similarity: input similarity matrix of shape
         [support example count, way * shot, query clip count, support clip count]
         :return: similarity matrix of shape
         [way * query_per_class, way , query clip count, support clip count * shot]
        """
        if self.args.video_to_class_matching == "joint":
            clip_to_clip_similarity = rearrange(
                clip_to_clip_similarity, "q (w s) lq ls -> q w lq (s ls)", s=self.args.shot)

        return clip_to_clip_similarity

    def get_video_to_video_similarity(
            self, clip_to_clip_similarity, clip_to_clip_similarity_transposed):
        """ Computes the video-to-video similarity score from the clip-to-clip similarity matrix
        and optionally its transposed version.

        :param clip_to_clip_similarity: clip-to-clip similarity matrix of shape
         [query video count, support video count, query clip count, support clip count]
        :param clip_to_clip_similarity_transposed: the transposed similarity matrix of shape
         [query video count, support video count, support clip count, query clip count]
         :return: the video-to-video similarity, as a tensor of shape
         [query video count, support video count]
        """
        if self.args.matching_function == "chamfer-support":
            video_to_video_similarity = self.matching_function(clip_to_clip_similarity_transposed)
        elif self.args.matching_function == "chamfer++":
            video_to_video_similarity = self.matching_function.forward(clip_to_clip_similarity)
            video_to_video_similarity_transposed = self.matching_function(
                clip_to_clip_similarity_transposed)
            video_to_video_similarity = 0.5 * (
                    video_to_video_similarity + video_to_video_similarity_transposed)
        else:
            video_to_video_similarity = self.matching_function(clip_to_clip_similarity)

        return video_to_video_similarity

    def forward(self, support_clips, support_labels, target_clips):
        """ Matching-based process composed of the following steps:
        - computes the clip-to-clip similarity matrix (clip-tuple to clip-tuple similarity matrix)
        - concatenates the support examples per class if video_to_class_matching=joint
        - transposes the clip-to-clip similarity matrix if needed
        - computes a video-to-video similarity score
        - computes a video-to-class similarity score if needed
        - multiply by a temperature factor (learnable)

        :param support_clips: the support clips if self.args.load_features is false / the support
          backbone embeddings if self.args.load_features is true
        :param support_labels: the labels of the support clips
        :param target_clips: the query clips if self.args.load_features is false / the query
          backbone embeddings if self.args.load_features is true
        :return: a dictionary with the video_to_class_similarity as a logit
        """
        clip_to_clip_similarity = self.get_clip_to_clip_similarity_matrix(
            support_clips, target_clips)

        clip_to_clip_similarity = self.concatenate_support_examples(clip_to_clip_similarity)

        clip_to_clip_similarity_transposed = self.transpose_similarity(clip_to_clip_similarity)

        clip_to_clip_similarity, clip_to_clip_similarity_transposed = self.filter_similarity(
            clip_to_clip_similarity, clip_to_clip_similarity_transposed)

        video_to_video_similarity = self.get_video_to_video_similarity(
            clip_to_clip_similarity, clip_to_clip_similarity_transposed)

        video_to_class_similarity = self.get_video_to_class_similarity(video_to_video_similarity)
        video_to_class_similarity *= self.temperature_weight  # learnt temperature weight
        video_to_class_similarity *= self.global_temperature  # fixed temperature

        return_dict = {'logits': video_to_class_similarity}

        return return_dict
