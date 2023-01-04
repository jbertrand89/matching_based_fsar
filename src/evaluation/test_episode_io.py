import os
import torch


def get_saved_episode_dir(args):
    """ Creates the directory containing the saved episodes based on the parameters in args.
    """

    saved_episodes_dir = os.path.join(
        args.test_episode_dir, args.dataset_name, "features",
        f"{args.dataset_name}_w{args.way}_s{args.shot}")
    print(f"10000 episodes in {saved_episodes_dir}")
    os.makedirs(saved_episodes_dir, exist_ok=True)
    return saved_episodes_dir


def save_episode(saved_episodes_dir, episode_id, task_dict):
    """ Saves for one episode:
    - the features of the support examples
    - the labels of the support examples
    - the features of the target/query examples
    - the labels of the target/query examples

    :param saved_episodes_dir: directory where the features will be saved
    :param episode_id: current episode number iteration
    :param task_dict: dictionary containing the support features, the support labels, the target
      features and the target labels
    """
    # save support features
    filename_support_set = os.path.join(saved_episodes_dir, f"support_set_{episode_id}.pth")
    torch.save(task_dict['support_set'], filename_support_set)

    # save support labels
    filename_support_labels = os.path.join(saved_episodes_dir, f"support_labels_{episode_id}.pth")
    torch.save(task_dict['support_labels'], filename_support_labels)

    # save target/query features
    filename_target_set = os.path.join(saved_episodes_dir, f"target_set_{episode_id}.pth")
    torch.save(task_dict['target_set'], filename_target_set)

    # save target/query labels
    filename_target_labels = os.path.join(saved_episodes_dir, f"target_labels_{episode_id}.pth")
    torch.save(task_dict['target_labels'], filename_target_labels)


def load_episode(saved_episodes_dir, episode_id):
    """ Loads for one episode:
    - the features of the support examples
    - the labels of the support examples
    - the features of the target/query examples
    - the labels of the target/query examples

    :param saved_episodes_dir: directory where the features will be saved
    :param episode_id: current episode number iteration
    :return: support features
    :return: support labels
    :return: target features
    :return: target labels
    """
    # load support features
    filename_support_set = os.path.join(saved_episodes_dir, f"support_set_{episode_id}.pth")
    support_features = torch.load(filename_support_set)

    # load support labels
    filename_support_labels = os.path.join(saved_episodes_dir, f"support_labels_{episode_id}.pth")
    support_labels = torch.load(filename_support_labels)

    # load target/query features
    filename_target_set = os.path.join(saved_episodes_dir, f"target_set_{episode_id}.pth")
    target_features = torch.load(filename_target_set)

    # load target/query labels
    filename_target_labels = os.path.join(saved_episodes_dir, f"target_labels_{episode_id}.pth")
    target_labels = torch.load(filename_target_labels)
    return support_features, support_labels, target_features, target_labels


def test_save_load(saved_episodes_dir, iteration, task_dict):
    """ Test that the saved features and labels are identical when saved and loader

    :param saved_episodes_dir: directory where the features will be saved
    :param episode_id: current episode number iteration
    :param task_dict: dictionary containing the support features, the support labels, the target
      features and the target labels
    """
    save_episodes(saved_episodes_dir, iteration, task_dict)
    support_set, support_labels, target_set, target_labels = load_episodes(
        saved_episodes_dir, iteration)

    ok = True
    if not torch.allclose(support_set, task_dict['support_set']):
        print(f"iteration {iteration} support set not equal")
        ok = False
    if not torch.allclose(target_set, task_dict['target_set']):
        print(f"iteration {iteration} target_set not equal")
        ok = False
    if not torch.allclose(support_labels, task_dict['support_labels']):
        print(f"iteration {iteration} support labels not equal")
        ok = False
    if not torch.allclose(target_labels, task_dict['target_labels']):
        print(f"iteration {iteration} target labels not equal")
        ok = False
    if ok:
        print("ok")

