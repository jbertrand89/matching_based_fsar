import argparse
import os
import time
import torch
from torch import nn
import torch.nn.parallel
import torch.optim as optim
from einops import rearrange

from src.evaluation.test_episode_io import load_episode, get_saved_episode_dir

import sys
path = os.path.abspath('../few-shot-video-classification')  # include the tsl repository
sys.path.append(path)
from tsl_fsv import weights_init, CLASSIFIER
from utils import setup_logger, AverageMeter, count_acc, euclidean_metric


""" The methods are adapted from 
https://github.com/xianyongqin/few-shot-video-classification/tsl_fsv r2+1d precomputed episodes.
"""


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
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=5)
    parser.add_argument('--test_way', type=int, default=5)
    parser.add_argument('--nepisode', type=int, default=500)
    parser.add_argument('--nepoch', default=5, type=int, help='Number of total epochs to run')
    parser.add_argument(
        '--lr', default=0.001, type=float,
        help='Initial learning rate (divided by 10 while training by lr scheduler)')
    parser.add_argument('--emb_dim', default=512, type=int, help='Temporal duration of inputs')
    parser.add_argument(
        '--n_val_samples', default=3, type=int, help='Number of validation clips for each video')
    parser.add_argument(
        '--result_dir', type=str, default=None, help='root directory where to save the results.')

    cmd_args = parser.parse_args()
    return cmd_args


def main_classifier_based_from_episodes():
    """ Runs the classifier-based approach on pre-saved episodes. First, loads episodes.
    For each episode, train a new classifier using the support examples, and
    evaluate the query examples with this classifier.
    """
    main_start_time = time.time()
    cmd_args = parse_command_line()

    if not os.path.exists(cmd_args.result_dir):
        os.makedirs(cmd_args.result_dir)

    # Setup logging system
    logger = setup_logger(
        "validation",
        cmd_args.result_dir,
        0,
        f'results_{cmd_args.manual_seed}_{cmd_args.shot}shots.txt'
    )
    logger.debug(cmd_args)

    torch.backends.cudnn.benchmark = True
    torch.manual_seed(cmd_args.manual_seed)

    episode_time = AverageMeter()
    accuracies = AverageMeter()
    full_episode_time = AverageMeter()

    start_sample = time.time()
    for i_episode in range(1, cmd_args.nepisode + 1):
        start_time = time.time()
        end_time = time.time()
        acc = meta_test_episode(cmd_args, i_episode)
        accuracies.update(acc)
        episode_time.update(time.time() - end_time)
        full_episode_time.update(time.time() - start_time)

        logger.info('Episode: {0}\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Full Time {full_time.val:.3f} ({full_time.avg:.3f})\t'
              'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                  i_episode,
                  batch_time=episode_time,
                  full_time=full_episode_time,
                  acc=accuracies))
        # raise Exception("debug")

    main_end_time = time.time()
    print(f"total time  {main_end_time - main_start_time}")
    print(f"time only episodes {main_end_time - start_sample}")


def meta_test_episode(cmd_args, episode_id):
    """ Trains a classifier on the support examples.

    :param cmd_args: command line parameters
    :param episode_id: episode id between 1 and 10000
    :return: accuracy value
    """
    # initialize a new classifier
    classifier = CLASSIFIER(cmd_args.emb_dim, cmd_args.test_way)
    classifier.apply(weights_init)
    criterion = nn.NLLLoss()
    classifier.cuda()
    criterion.cuda()
    optimizer = optim.Adam(classifier.parameters(), lr=cmd_args.lr, betas=(0.5, 0.999))

    # loads the episode
    support_features, support_labels, target_features, target_labels = load_episode_and_format(
        cmd_args, episode_id)

    # trains the classifier
    for _ in range(cmd_args.nepoch):
        train_epoch_from_features(
            classifier, criterion, optimizer, support_features, support_labels)

    # evaluates
    acc = val_epoch_from_features(classifier, target_features, target_labels, cmd_args)
    return acc


def load_episode_and_format(cmd_args, episode_id):
    """ Trains a classifier on the support examples.

    :param cmd_args: command line parameters
    :param episode_id: episode id between 1 and 10000
    :return: R2+1D features of the support examples
    :return: labels of the support examples
    :return: R2+1D features of the target/query examples
    :return: labels of the target/query examples
    """
    cmd_args.test_episode_dir = "/mnt/personal/bertrjul/debug_github/episodes/"

    saved_episodes_dir = get_saved_episode_dir(
        cmd_args.test_episode_dir, cmd_args.dataset, cmd_args.test_way, cmd_args.shot)

    support_features, support_labels, target_features, target_labels = load_episode(
        saved_episodes_dir, episode_id)

    support_features = rearrange(support_features, "a b c -> (a b) c")
    support_labels = support_labels.type(torch.LongTensor)
    support_labels = support_labels.to(support_features.device)
    support_labels = torch.repeat_interleave(support_labels, cmd_args.n_val_samples)

    target_features = rearrange(target_features, "a b c -> (a b) c")
    target_labels = target_labels.type(torch.LongTensor)
    target_labels = target_labels.to(target_features.device)
    return support_features, support_labels, target_features, target_labels


def train_epoch_from_features(classifier, criterion, optimizer, support_features, support_labels):
    """ Trains a classifier on the support examples.

    :param classifier: classifier to be trained
    :param criterion:
    :param optimizer:
    :param support_features: R2+1D features of the support examples
    :param support_labels: labels of the support examples
    """
    classifier.train()
    optimizer.zero_grad()
    output = classifier(support_features)
    loss = criterion(output, support_labels)
    loss.backward()
    optimizer.step()


def val_epoch_from_features(classifier, target_features, target_labels, args):
    """Classifies the target/query examples using the classifier.

    :param classifier: classifier
    :param target_features: R2+1D features of the target/query examples
    :param target_labels: labels of the target/query examples
    :param args: command line parameters
    :return: accuracy value
    """
    classifier.eval()
    with torch.no_grad():
        clip_logits = torch.exp(classifier(target_features))
        logits = clip_logits.reshape(args.query * args.test_way, args.n_val_samples, -1).mean(dim=1)
        acc, pred = count_acc(logits, target_labels)
    return acc


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main_classifier_based_from_episodes()
