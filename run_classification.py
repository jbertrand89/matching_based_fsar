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
from opts import parse_opts
from mean import get_mean, get_std
from utils import setup_logger, AverageMeter, count_acc, euclidean_metric


def main_load_features():
    main_start_time = time.time()
    opt = parse_opts()
    print(opt)

    opt.scales = [opt.initial_scale]
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)

    opt.arch = '{}-{}'.format(opt.clip_model, opt.clip_model_depth)
    opt.mean = get_mean(opt.norm_value)
    opt.std = get_std(opt.norm_value)

    if not os.path.exists(opt.result_path):
        os.makedirs(opt.result_path)

    # Setup logging system
    logger = setup_logger(
        "validation",
        opt.result_path,
        0,
        'results.txt'
    )
    logger.debug(opt)
    print(opt.lr)
    if opt.gpu is not None:
        print("Use GPU: {} for training".format(opt.gpu))

    torch.backends.cudnn.benchmark = True

    torch.manual_seed(opt.manual_seed)

    episode_time = AverageMeter()
    accuracies = AverageMeter()
    full_episode_time = AverageMeter()

    start_sample = time.time()
    for i_episode in range(1, opt.nepisode + 1):
        start_time = time.time()
        end_time = time.time()
        acc = meta_test_episode(opt, i_episode)
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


def meta_test_episode(args, episode_id):
    """ Trains a classifier on the support examples.

    :param args: command line parameters
    :param episode_id: episode id between 1 and 10000
    :return: accuracy value
    """
    # initialize a new classifier
    classifier = CLASSIFIER(args.emb_dim, args.test_way)
    classifier.apply(weights_init)
    criterion = nn.NLLLoss()
    classifier.cuda()
    criterion.cuda()
    optimizer = optim.Adam(classifier.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # loads the episode
    support_features, support_labels, target_features, target_labels = load_episode_and_format(
        args, episode_id)

    # trains the classifier
    for _ in range(args.nepoch):
        train_epoch_from_features(
            classifier, criterion, optimizer, support_features, support_labels)

    # evaluates
    acc = val_epoch_from_features(classifier, target_features, target_labels, args)
    return acc


def load_episode_and_format(args, episode_id):
    """ Trains a classifier on the support examples.

    :param args: command line parameters
    :param episode_id: episode id between 1 and 10000
    :return: R2+1D features of the support examples
    :return: labels of the support examples
    :return: R2+1D features of the target/query examples
    :return: labels of the target/query examples
    """
    args.test_episode_dir = "/mnt/personal/bertrjul/debug_github/episodes/"
    args.dataset_name = args.dataset
    args.way = args.test_way

    saved_episodes_dir = get_saved_episode_dir(args)
    #args.test_episode_dir, args.dataset_name, f"{args.dataset_name}_w{args.way}_s{args.shot}")

    support_features, support_labels, target_features, target_labels = load_episode(
        saved_episodes_dir, episode_id)

    support_features = rearrange(support_features, "a b c -> (a b) c")
    support_labels = support_labels.type(torch.LongTensor)
    support_labels = support_labels.to(support_features.device)
    support_labels = torch.repeat_interleave(support_labels, args.n_val_samples)

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
    main_load_features()
