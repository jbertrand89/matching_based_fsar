import argparse
import os
import numpy as np


def parse_command_line():
    """ Parses the command line.

    :return: command line parameters
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--result_dir', type=str, default=None, help='Directory where the results are saved')
    parser.add_argument(
        '--result_template', type=str, default=None, help='Start of the result file')
    parser.add_argument(
        '--result_template_end', type=str, default="_log_test_10000", help='End of the result file')
    parser.add_argument(
        '--using_classifier',  default=False, action="store_true")
    parser.add_argument(
        '--seeds', nargs='+', type=int, help='model seeds', default=[1, 5, 10])
    cmd_args = parser.parse_args()
    return cmd_args


def read_accuracy(filename, using_classifier):
    """ Loads the filename and extract the accuracy value from it

    :param filename: result filename
    :param using_classifier: whether the filename was created from the classifier-based or
      matching-based methods
    :return: accuracy
    """
    if using_classifier:
        return read_accuracy_classifier_based(filename)
    else:
        return read_accuracy_matching_based(filename)


def read_accuracy_matching_based(filename):
    """ Loads the filename and extract the accuracy value from it

    :param filename: result filename
    :return: accuracy
    """
    with open(filename, "r+") as reader:
        data = reader.readlines()

    for i in range(len(data)):
        line = data[i].strip()
        if line == "test accuracies:":
            next_line = data[i + 1].strip()
            return float((next_line.split(":")[-1]).split("+")[0])


def read_accuracy_classifier_based(filename):
    """ Loads the filename and extract the accuracy value from it

    :param filename: result filename
    :return: accuracy
    """
    with open(filename, "r+") as reader:
        data = reader.readlines()

    line = data[-1].strip()
    return 100*float((line.split("(")[-1]).split(")")[0])


def average_multi_seeds(
        result_dir, file_template_start, file_template_end, seeds, using_classifier):
    """ Loads the filename and extract the accuracy value from it

    :param result_dir: directory where the results are saved
    :param file_template_start: template for the result files
    :param file_template_end: template for the result files
    :param seeds: list of model seeds
    :param using_classifier: whether the filename was created from the classifier-based or
      matching-based methods
    """
    accuracies = []
    for seed in seeds:
        filename = os.path.join(result_dir, f"{file_template_start}{seed}{file_template_end}.txt")
        accuracies.append(read_accuracy(filename, using_classifier))

    print(accuracies)
    mean = np.mean(accuracies)
    std = np.std(accuracies)
    print(f"accuracy {mean:.1f} +- {std:.1f}")


if __name__ == "__main__":
    args = parse_command_line()
    average_multi_seeds(
        args.result_dir, args.result_template, args.result_template_end, args.seeds,
        args.using_classifier)
