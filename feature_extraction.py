import argparse
import numpy as np
import os
import time
import torch

import sys
path = os.path.abspath('../few-shot-video-classification')  # include the tsl repository
sys.path.append(path)

from src.feature_extraction_utils.spatial_transformations import get_spatial_transorm
from src.feature_extraction_utils.video_clips import load_clips, get_clip_embeddings
from models import r2plus1d


def parse_command_line():
    """ Parses the command line.

    :return: command line parameters
    """
    parser = argparse.ArgumentParser()

    # GENERAL
    parser.add_argument(
        '--manual_seed', default=1, type=int, help='Manually set random seed')
    parser.add_argument('--dataset', type=str)
    parser.add_argument(
        '--batch_size', default=32, type=int, help='Batch size for computing the embeddings')
    parser.add_argument(
        '--n_threads', default=4, type=int, help='Number of threads for multi-thread loading')
    parser.add_argument('--num_gpus', type=int, default=1)

    # CLIP
    parser.add_argument('--image_size', default=112, type=int, help='Height and width of inputs')
    parser.add_argument(
        '--clip_length', default=16, type=int, help='Temporal duration of inputs')

    # SPATIAL TRANSFORMS
    parser.add_argument(
        '--norm_value', default=1, type=int,
        help='If 1, range of inputs is [0-255]. If 255, range of inputs is [0-1].')
    parser.add_argument(
        '--train_crop', default='center', type=str,
        help='Spatial cropping method in training ( random | corner | center)')
    parser.add_argument(
        '--no_mean_norm', action='store_true', help='If true, inputs are not normalized by mean.')
    parser.add_argument(
        '--std_norm', action='store_true',
        help='If true, inputs are normalized by standard deviation.')
    parser.add_argument(
        '--initial_scale', default=1.0, type=float, help='Initial scale for multiscale cropping')
    parser.add_argument(
        '--n_scales', default=5, type=int, help='Number of scales for multiscale cropping')
    parser.add_argument(
        '--scale_step', default=0.84089641525, type=float,
        help='Scale step for multiscale cropping')
    parser.add_argument(
        '--spatial_transform_horizontal_flip', default=False, action='store_true',
        help='If true, apply horizontal flip while training.')

    # MODEL
    parser.add_argument(
        '--pretrain_path', default='', type=str, help='Pretrained model (.pth)')
    parser.add_argument(
        '--r2plus1d_n_classes_pretrain', default=64, type=int,
        help='Number of classes the pretrained model was trained on'
    )

    # DIRECTORIES
    parser.add_argument(
        '--input_video_dir', type=str, default=None,
        help='Video directory to be extracted. If not None, it will only extract the features of '
             'the video. Otherwise, you need to define input_dataset_dir.')
    parser.add_argument(
        '--output_video_dir', type=str, default=None,
        help='Directory where to save the features.')
    parser.add_argument(
        '--input_dataset_dir', type=str, default=None,
        help='Dataset root directory. Must be defined if input_video_dir is None.')
    parser.add_argument(
        '--output_dataset_dir', type=str, default=None,
        help='Features root directory. Must be defined if you extract features for a dataset')
    parser.add_argument('--split', type=str, help='Dataset split name')
    parser.add_argument(
        '--class_ids', type=str, nargs='+', default=None,
        help='Class ids of the videos you want to extract features for. If None, all the classes '
             'will be processed')
    parser.add_argument('--log_dir', type=str, default=None)

    cmd_args = parser.parse_args()
    return cmd_args


def get_paths(cmd_args):
    """
    Get the source paths of the videos to be extracted and their corresponding destination paths.
    If a video directory is provided (cmd_args.input_video_dir) only process this one, otherwise process
    the dataset split.

    :param cmd_args: command line parameters
    :return: list of tuples of all the videos to be embedded. Each feature contains the input video
       directory and the output feature directory.
    """
    #
    if cmd_args.input_video_dir:
        os.makedirs(cmd_args.output_video_dir, exist_ok=True)
        paths = [(cmd_args.input_video_dir, cmd_args.output_video_dir)]
    elif cmd_args.input_dataset_dir is None:
        raise Exception("You need to define input_video_dir or input_dataset_dir")
    else:
        paths = get_dataset_paths(cmd_args)

    return paths


def get_dataset_paths(cmd_args):
    """
    Get the source paths of the videos to be extracted and their corresponding destination paths
    for an entire dataset split, specified with cmd_args.input_dataset_dir and cmd_args.split.

    :param cmd_args: command line parameters
    :return: list of tuples of all the videos to be embedded. Each feature contains the input video
       directory and the output feature directory.
    """
    paths = []
    output_split_dir = os.path.join(cmd_args.output_dataset_dir, cmd_args.split)
    input_split_dir = os.path.join(cmd_args.input_dataset_dir, cmd_args.split)

    # get the class names corresponding to the class_ids. If class_ids is None, get all the
    # classes available
    if cmd_args.class_ids is None:
        class_names = os.listdir(input_split_dir)
    else:
        if cmd_args.dataset in {"kinetics", "ucf101"}:
            all_class_names = np.array(sorted(os.listdir(input_split_dir)))
            class_names = [all_class_names[int(class_id)] for class_id in cmd_args.class_ids]
            # ids = [int(class_id) for class_id in cmd_args.class_ids]
            # class_names = [f"{class_folder}" for class_folder in all_class_names[ids] ]
        else:
            class_names = [f"{cmd_args.split}{class_id}" for class_id in cmd_args.class_ids]

    for class_name in class_names:
        output_class_dir = os.path.join(output_split_dir, class_name)
        input_class_dir = os.path.join(input_split_dir, class_name)

        if not os.path.isdir(input_class_dir):
            continue

        for video_name in os.listdir(input_class_dir):
            output_video_dir = os.path.join(output_class_dir, video_name)
            input_video_dir = os.path.join(input_class_dir, video_name)
            os.makedirs(output_video_dir, exist_ok=True)
            paths.append((input_video_dir, output_video_dir))
    return paths


def get_model(cmd_args):
    """ Loads the r2+1d backbone.

    :param cmd_args: command line parameters
    :return: backbone model
    """
    model = r2plus1d.r2plus1d_34(num_classes=cmd_args.r2plus1d_n_classes_pretrain)
    model = torch.nn.DataParallel(model, device_ids=[i for i in range(cmd_args.num_gpus)])

    # Load pretrained model
    if cmd_args.pretrain_path:
        pretrain = torch.load(cmd_args.pretrain_path)
        model.load_state_dict(pretrain['state_dict'])

    # Remove last layer
    model = torch.nn.Sequential(*list(model.module.children())[:-1])

    model = torch.nn.DataParallel(model, device_ids=[i for i in range(cmd_args.num_gpus)])

    return model


def extract_features_from_paths(cmd_args, paths, spatial_transform, model):
    """ Extracts the features for all the videos defined in paths

    :param cmd_args: command line parameters
    :param paths: list of tuples of all the videos to be embedded. Each feature contains the input
      video directory and the output feature directory.
    :param spatial_transform: spatial transformations to be applied to each frame of the clip
    :param model: backbone model
    :return: list of tuples containing the path of the video processed and if the extraction was
      successful
    """
    processed = []
    for input_video_dir, output_video_dir in paths:
        try:
            extract_video_features(
                input_video_dir, output_video_dir, spatial_transform, cmd_args.clip_length, model)
            processed.append((input_video_dir, True))
        except Exception as e:
            print(e)
            processed.append((input_video_dir, False))
    return processed


def save_clip_frame_names(output_dir, clip_frame_names):
    """ Extracts the video features for the given video.

    :param output_dir: directory where to save the clip frame names
    :param clip_frame_names: list of all the frame names in the clips clips
    """
    clip_names_filename = os.path.join(output_dir, "clip_names.txt")
    with open(clip_names_filename, "w+") as writer:
        for clip_names in clip_frame_names:
            writer.write(",".join(clip_names))
            writer.write("\n")


def extract_video_features(video_dir, output_dir, spatial_transform, clip_length, model):
    """ Extracts the video features for the given video.

    :param video_dir: directory containing the video frames
    :param output_dir: directory where to save the features
    :param spatial_transform: spatial transformations to be applied to each frame of the clip
    :param clip_length: number of consecutive frames in a clip
    :param model: backbone model
    :return: list of tuples containing the path of the video processed and if the extraction was
      successful
    """
    # Loads the clips
    clips, clip_frame_names = load_clips(video_dir, spatial_transform, clip_length)

    # Save the clip frame names
    save_clip_frame_names(output_dir, clip_frame_names)

    # Apply the model
    with torch.no_grad():
        features = get_clip_embeddings(clips, model, cmd_args.batch_size)

    # Save the features
    feature_filename = os.path.join(output_dir, "video_features.pth")
    torch.save(features, feature_filename)


def save_logs(cmd_args, processed):
    """ Saves the processed logs in a file, to track if there are issues.

    :param cmd_args: command line parameters
    :return: list of tuples containing the video processed and if the extraction was successful
    """
    # filename
    if cmd_args.input_video_dir:
        log_filename = os.path.join(cmd_args.log_dir, f"processed.txt")
    else:
        class_ids_text = '_'.join(cmd_args.class_ids) if cmd_args.class_ids else "all"
        log_filename = os.path.join(cmd_args.log_dir, f"processed_{cmd_args.split}_{class_ids_text}.txt")

    with open(log_filename, "w+") as writer:
        for p in processed:
            writer.write(";".join([str(i) for i in p]) + "\n")
    print(f"saved {log_filename} ")


if __name__ == '__main__':
    # Parse command line
    cmd_args = parse_command_line()

    # Fix seeds
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(cmd_args.manual_seed)
    from torch.backends import cudnn
    cudnn.deterministic = True  # type: ignore
    cudnn.benchmark = False  # type: ignore

    # Get the spatial transformations
    spatial_transform = get_spatial_transorm(cmd_args)

    # Model
    model = get_model(cmd_args)
    model.eval()

    # Get the video paths
    paths = get_paths(cmd_args)

    # Extract features
    t0 = time.time()
    processed = extract_features_from_paths(cmd_args, paths, spatial_transform, model)
    t1 = time.time()
    print(f"extracted in  {t1 - t0} ")

    # Saving logs
    if cmd_args.log_dir:
        save_logs(cmd_args, processed)
