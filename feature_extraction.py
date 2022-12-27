import argparse
import numpy as np
import os
import time
import torch

import sys
path = os.path.abspath('../few-shot-video-classification')  # include the tsl repository
sys.path.append(path)

from src.spatial_transformations import get_spatial_transorm
from src.video_clips import load_clips, get_clip_embeddings
from models import r2plus1d


def parse_command_line():
    parser = argparse.ArgumentParser()

    # Global parameters
    parser.add_argument(
        '--manual_seed', default=1, type=int, help='Manually set random seed')
    parser.add_argument('--dataset', type=str)
    parser.add_argument(
        '--batch_size', default=32, type=int, help='Batch size for computing the embeddings')
    parser.add_argument(
        '--n_threads', default=4, type=int, help='Number of threads for multi-thread loading')
    parser.add_argument('--num_gpus', type=int, default=1)

    # Clip dimensions
    parser.add_argument('--image_size', default=112, type=int, help='Height and width of inputs')
    parser.add_argument(
        '--clip_length', default=16, type=int, help='Temporal duration of inputs')

    # Spatial transforms
    parser.add_argument(
        '--norm_value', default=1, type=int,
        help='If 1, range of inputs is [0-255]. If 255, range of inputs is [0-1].')
    parser.add_argument(
        '--train_crop', default='corner', type=str,
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
    parser.add_argument('--spatial_transform_horizontal_flip', default=True, action='store_false',
                        help='If true, apply horizontal flip while training.')

    # MODEL
    parser.add_argument(
        '--pretrain_path', default='', type=str, help='Pretrained model (.pth)')
    parser.add_argument(
        '--r2plus1d_n_classes_pretrain', default=64, type=int,
        help='Number of classes the pretrained model was trained on'
    )

    # Directories
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

    args = parser.parse_args()
    return args


def get_paths(opt):
    """
    Get the source paths of the videos to be extracted and their corresponding destination paths.
    If a video directory is provided (opt.input_video_dir) only process this one, otherwise process
    the dataset split.
    """
    #
    if opt.input_video_dir:
        os.makedirs(opt.output_video_dir, exist_ok=True)
        paths = [[opt.input_video_dir, opt.output_video_dir]]
    elif opt.input_dataset_dir is None:
        raise Exception("You need to define input_video_dir or input_dataset_dir")
    else:
        paths = get_dataset_paths()

    return paths


def get_dataset_paths():
    """
    Get the source paths of the videos to be extracted and their corresponding destination paths
    for an entire dataset split, specified with opt.input_dataset_dir and opt.split.
    """
    paths = []
    output_split_dir = os.path.join(opt.output_dataset_dir, opt.split)
    input_split_dir = os.path.join(opt.input_dataset_dir, opt.split)

    # get the class names corresponding to the class_ids. If class_ids is None, get all the
    # classes available
    if opt.class_ids is None:
        class_names = os.listdir(input_split_dir)
    else:
        if opt.dataset in {"kinetics", "ucf101"}:
            all_folders = np.array(os.listdir(input_split_dir))
            ids = [int(cid) for cid in opt.class_ids]
            class_names = [f"{class_folder}" for class_folder in all_folders[ids]]
        else:
            class_names = [f"{opt.split}{class_folder}" for class_folder in opt.class_ids]

    for class_name in class_names:
        output_class_dir = os.path.join(output_split_dir, class_name)
        input_class_dir = os.path.join(input_split_dir, class_name)

        if not os.path.isdir(input_class_dir):
            continue

        for video_name in os.listdir(input_class_dir):
            output_video_dir = os.path.join(output_class_dir, video_name)
            input_video_dir = os.path.join(input_class_dir, video_name)
            os.makedirs(output_video_dir, exist_ok=True)
            paths.append([input_video_dir, output_video_dir])
    return paths


def get_model(opt):
    model = r2plus1d.r2plus1d_34(num_classes=opt.r2plus1d_n_classes_pretrain)
    model = torch.nn.DataParallel(model, device_ids=[i for i in range(opt.num_gpus)])

    # Load pretrained model
    if opt.pretrain_path:
        pretrain = torch.load(opt.pretrain_path)
        model.load_state_dict(pretrain['state_dict'])

    # Remove last layer
    model = torch.nn.Sequential(*list(model.module.children())[:-1])

    model = torch.nn.DataParallel(model, device_ids=[i for i in range(opt.num_gpus)])

    return model


def extract_features(opt, paths, spatial_transform, model):
    processed = []
    for input_video_dir, output_video_dir in paths:
        try:
            extract_video_features(
                spatial_transform, input_video_dir, opt.clip_length, output_video_dir, model)
            processed.append([input_video_dir, True])
        except Exception as e:
            print(e)
            processed.append([input_video_dir, False])
    return processed


def extract_video_features(spatial_transform, video_path, clip_length, output_folder, model):
    # Loads the clips
    clips = load_clips(video_path, spatial_transform, clip_length)

    # Apply the model
    with torch.no_grad():
        features = get_clip_embeddings(clips, model, opt.batch_size)

    # Save the features
    feature_filename = os.path.join(output_folder, "video_features.pth")
    torch.save(features, feature_filename)


def save_logs(opt, processed):
    # filename
    if opt.input_video_dir:
        log_filename = os.path.join(opt.log_dir, f"processed.txt")
    else:
        class_ids_text = '_'.join(opt.class_ids) if opt.class_ids else "all"
        log_filename = os.path.join(opt.log_dir, f"processed_{opt.split}_{class_ids_text}.txt")

    with open(log_filename, "w+") as writer:
        for p in processed:
            writer.write(";".join([str(i) for i in p]) + "\n")
    print(f"saved {log_filename} ")


if __name__ == '__main__':
    # Parse command line
    opt = parse_command_line()

    # Fix seeds
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(opt.manual_seed)
    from torch.backends import cudnn

    cudnn.deterministic = True  # type: ignore
    cudnn.benchmark = False  # type: ignore

    # Get the spatial transformations
    spatial_transform = get_spatial_transorm(opt)

    # Model
    model = get_model(opt)
    model.eval()

    # Get the video paths
    paths = get_paths(opt)

    # Extract features
    t0 = time.time()
    processed = extract_features(opt, paths, spatial_transform, model)
    t1 = time.time()
    print(f"extracted in  {t1 - t0} ")

    # Saving logs
    if opt.log_dir:
        save_logs(opt, processed)
