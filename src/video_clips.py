import os
import torch
from PIL import Image


def pil_loader(filename):
    """Loads the image."""
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(filename, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def load_clip_frames(video_dir, clip_length):
    """Loads all the possible video clips of length clip_length from the video directory."""
    frame_count = len(os.listdir(video_dir))
    max_start_index = frame_count - clip_length + 1

    clips = []
    if frame_count <= clip_length:  # edge case: when there are less frames than the minimum clip
        # length, repeat the last frame
        clip = [pil_loader(os.path.join(video_dir, f"{frame_id:08}.jpg"))
                for frame_id in range(1, frame_count + 1)]
        while len(clip) < clip_length:
            clip.append(clip[-1])
        clips.append(clip)
    else:
        for start in range(1, max_start_index + 1):
            clip = [pil_loader(os.path.join(video_dir, f"{frame_id:08}.jpg"))
                    for frame_id in range(start, start + clip_length)]
            clips.append(clip)
    return clips


def load_clips(video_dir, spatial_transform, clip_length):
    """
    Loads all the possible video clips from the video directory, with spatial transformations
    (cropping, flipping, normalization) applied similarly to all the frames of a clip.
    """
    # load the clips
    clips = load_clip_frames(video_dir, clip_length)

    # apply spatial transform and stack
    spatial_transform.randomize_parameters()
    transformed_clips = []
    for clip in clips:
        transformed_clip = [spatial_transform(img) for img in clip]
        transformed_clip = torch.stack(transformed_clip, 0).permute(1, 0, 2, 3)
        transformed_clips.append(transformed_clip)

    transformed_clips = torch.stack(transformed_clips, dim=0)

    return transformed_clips


def get_clip_embeddings(clips, model, batch_size):
    """Extracts the clip features using the given model."""
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
