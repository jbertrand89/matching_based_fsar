import os
import torch
from PIL import Image


def pil_loader(filename: str):
    """Loads the image.

    :param filename: image filename
    :return: RGB image
    """
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(filename, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def load_clip_frames(video_dir: str, clip_length: int):
    """Loads all the possible video clips of length clip_length from the video directory.

    :param video_dir: path of the directory containing the video frames
    :param clip_length: number of consecutive frames in a clip
    :return: list of all the possible clips
    :return: list of all the frame names in the clips clips
    """
    frame_count = len(os.listdir(video_dir))
    max_start_index = frame_count - clip_length + 1

    clips = []
    clip_frame_names = []
    clip_frame_ids = []
    if frame_count <= clip_length:  # edge case: when there are less frames than the minimum clip
        # length, repeat the last frame
        frame_ids = list(range(1, frame_count + 1))
        while len(frame_ids) < clip_length:
            frame_ids.append(frame_ids[-1])
        clip_frame_ids.append(frame_ids)
    else:
        for start in range(1, max_start_index + 1):
            frame_ids = list(range(start, start + clip_length))
            clip_frame_ids.append(frame_ids)

    for frame_ids in clip_frame_ids:
        clip = [pil_loader(os.path.join(video_dir, f"{frame_id:08}.jpg"))
                for frame_id in frame_ids]
        clips.append(clip)

        frame_names = [f"{frame_id:08}.jpg" for frame_id in frame_ids]
        clip_frame_names.append(frame_names)
    return clips, clip_frame_names


def load_clips(video_dir: str, spatial_transform, clip_length: int):
    """
    Loads all the possible video clips from the video directory, with spatial transformations
    (cropping, flipping, normalization) applied similarly to all the frames of a clip.

    :param video_dir: path of the directory containing the video frames
    :param spatial_transform: spatial transformations to be applied to each frame of the clip
    :param clip_length: number of consecutive frames in a clip
    :return: tensor containing all the possible clips after spatial transformation
    :return: list of all the frame names in the clips clips
    """
    # load the clips
    clips, clip_frame_names = load_clip_frames(video_dir, clip_length)

    # apply spatial transform and stack
    spatial_transform.randomize_parameters()
    transformed_clips = []
    for clip in clips:
        transformed_clip = [spatial_transform(img) for img in clip]
        transformed_clip = torch.stack(transformed_clip, 0).permute(1, 0, 2, 3)
        transformed_clips.append(transformed_clip)

    transformed_clips = torch.stack(transformed_clips, dim=0)

    return transformed_clips, clip_frame_names


@torch.no_grad()
def get_clip_embeddings(clips, model, batch_size: int):
    """Extracts the clip features using the given model.

    :param clips: input clips from which the features will be extracted
    :param model: backbone model
    :param batch_size: batch size to extract the embeddings
    :return: the clip embeddings
    """
    clip_embedding = []
    cur_loc = 0
    while cur_loc + batch_size < clips.shape[0]:
        batch_clips = clips[cur_loc:cur_loc + batch_size]
        batch_embedding = model(batch_clips).squeeze()
        clip_embedding.append(batch_embedding)
        cur_loc += batch_size

    # Final batch
    batch_clips = clips[cur_loc:]
    batch_embedding = model(batch_clips).squeeze()
    if batch_clips.shape[0] == 1:  # if there is one element only in the batch, unsqueeze it to
        # match the other batches
        batch_embedding = batch_embedding.unsqueeze(dim=0)
    clip_embedding.append(batch_embedding)

    clip_embedding = torch.cat(clip_embedding, dim=0)
    return clip_embedding
