import time
import torch
import numpy as np
import argparse
import os
import random
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter

import sys
path = os.path.abspath('../few-shot-action-recognition')  # include the trx repository
sys.path.append(path)
from utils import print_and_log, get_log_files, TestAccuracies, aggregate_accuracy, verify_checkpoint_dir

from src.few_shot_models import TRX_few_shot_model, MatchingBasedFewShotModel
from src.data_loaders.video_feature_reader import VideoFeatureReader
from src.evaluation.test_episode_io import save_episode, load_episode, get_saved_episode_dir


# torch.autograd.set_detect_anomaly(True)

class Learner:
    def __init__(self):
        self.args = self.parse_command_line()

        if self.args.seed:
            torch.manual_seed(self.args.seed)
            torch.cuda.manual_seed_all(self.args.seed)
            np.random.seed(self.args.seed)
            random.seed(self.args.seed)

            # from torch.backends import cudnn
            #
            # cudnn.deterministic = True  # type: ignore
            # cudnn.benchmark = False  # type: ignore

        self.checkpoint_dir, self.logfile, self.checkpoint_path_validation, self.checkpoint_path_final \
            = get_log_files(self.args.checkpoint_dir, self.args.resume_from_checkpoint, False)
        self.args.logfile = self.logfile

        print_and_log(self.logfile, "Options: %s\n" % self.args)
        print_and_log(self.logfile, "Checkpoint Directory: %s\n" % self.checkpoint_dir)

        self.writer = SummaryWriter()
        self.device = torch.device(
            'cuda' if (torch.cuda.is_available() and self.args.num_gpus > 0) else 'cpu')
        self.args.device = self.device
        self.model = self.init_model()

        if not self.args.load_test_episodes or not self.args.test_only:
            self.video_dataset = VideoFeatureReader(self.args)
            self.video_loader = torch.utils.data.DataLoader(
                self.video_dataset, batch_size=1, num_workers=self.args.num_workers)

        self.val_accuracies = TestAccuracies([self.args.dataset]) # todo check this

        self.accuracy_fn = aggregate_accuracy

        if not self.args.test_only:
            if self.args.opt == "adam":
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
            elif self.args.opt == "sgd":
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.learning_rate)
        
            self.scheduler = MultiStepLR(self.optimizer, milestones=self.args.sch, gamma=0.1)
            self.optimizer.zero_grad()
        
        self.start_iteration = 0

    def init_model(self):
        if self.args.method == "trx":
            model = TRX_few_shot_model(self.args)
        elif self.args.method == "matching-based":
            model = MatchingBasedFewShotModel(self.args)

        model = model.to(self.device) 

        if torch.cuda.is_available() and self.args.num_gpus > 1:
            model.distribute_model()
        return model

    def parse_command_line(self):
        parser = argparse.ArgumentParser()

        parser.add_argument("--dataset", type=str, default="data/ssv2small", help="Path to dataset")
        parser.add_argument("--learning_rate", "-lr", type=float, default=0.001, help="Learning rate.")
        parser.add_argument("--tasks_per_batch", type=int, default=16, help="Number of tasks between parameter optimizations.")
        parser.add_argument("--checkpoint_dir", "-c", default=None, help="Directory to save checkpoint to.")
        parser.add_argument("--test_model_name", "-m", default=None, help="Path to model to load and test.")
        parser.add_argument("--training_iterations", "-i", type=int, default=60000, help="Number of meta-training iterations.")
        parser.add_argument("--resume_from_checkpoint", "-r", dest="resume_from_checkpoint", default=False, action="store_true", help="Restart from latest checkpoint.")
        parser.add_argument("--way", type=int, default=5, help="Way of each task.")
        parser.add_argument("--shot", type=int, default=5, help="Shots per class.")
        parser.add_argument("--query_per_class", "-qpc", type=int, default=5, help="Target samples (i.e. queries) per class used for training.")
        parser.add_argument("--query_per_class_test", "-qpct", type=int, default=1, help="Target samples (i.e. queries) per class used for testing.")
        parser.add_argument('--val_iters', nargs='+', type=int, help='iterations to val at.', default=[])
        parser.add_argument("--num_val_tasks", type=int, default=1000, help="number of random tasks to val on.")
        parser.add_argument("--num_test_tasks", type=int, default=10000, help="number of random tasks to test on.")
        parser.add_argument("--print_freq", type=int, default=1000, help="print and log every n iterations.")
        parser.add_argument("--seq_len", type=int, default=8, help="Frames per video.")
        parser.add_argument("--num_workers", type=int, default=8, help="Num dataloader workers.")
        parser.add_argument(
            "--backbone", choices=["resnet18", "resnet34", "resnet50", "r2+1d", "r2+1d_fc"],
            default="r2+1d", help="backbone")
        parser.add_argument("--opt", choices=["adam", "sgd"], default="sgd", help="Optimizer")
        parser.add_argument("--save_freq", type=int, default=5000, help="Number of iterations between checkpoint saves.")
        parser.add_argument("--img_size", type=int, default=224, help="Input image size to the CNN after cropping.")
        parser.add_argument("--num_gpus", type=int, default=4, help="Number of GPUs to split the ResNet over")
        parser.add_argument('--sch', nargs='+', type=int, help='iters to drop learning rate', default=[1000000])
        parser.add_argument("--method", choices=["trx", "matching-based", "tsn", "pal"], default="trx", help="few-shot method to use")
        parser.add_argument("--pretrained_backbone", "-pt", type=str, default=None, help="pretrained backbone path, used by PAL")
        parser.add_argument("--val_on_test", default=False, action="store_true", help="Danger: Validate on the test set, not the validation set. Use for debugging or checking overfitting on test set. Not good practice to use when developing, hyperparameter tuning or training models.")

        # PARAMETERS ADDED for the paper Rethinking matching-based few-shot action recognition
        parser.add_argument("--dataset_name", type=str)
        parser.add_argument('--seed', type=int, default=1, help="global seed value")
        parser.add_argument(
            '--temp_set', nargs='+', type=int, help='cardinalities e.g. 2,3 is pairs and triples',
            default=[2, 3])
        parser.add_argument("--gradient_clip", type=int, default=1000)

        # dataloader parameters
        parser.add_argument(
            "--split_paths", nargs='+', default=None,
            help="split paths, used in the feature loader.")
        parser.add_argument(
            "--split_names", nargs='+', default=None,
            help="split names, used in the feature loader.")
        parser.add_argument(
            "--split_seeds", nargs='+', default=None, help="generator seeds")

        # evaluation parameters
        parser.add_argument("--evaluation_mode", choices=["test", "val"], default="test",
            help="run evaluation on test or val")
        parser.add_argument(
            '--get_best_val_checkpoint', default=False, action="store_true",
            help="run the evaluation on the best model evaluated on the validation dataset")
        parser.add_argument(
            "--save_test_episodes", default=False, action="store_true",
            help="if set to true, it will save all the test episodes in test_episode_dir")
        parser.add_argument(
            "--load_test_episodes", default=False, action="store_true",
            help="if set to true, it will load all the test episodes from test_episode_dir")
        parser.add_argument(
            "--test_episode_dir", type=str, default="",
            help="needs to be defined if save_test_episodes or load_test_episodes is true")

        # matching temperatures
        parser.add_argument(
            '--matching_global_temperature', type=float, default=10,
            help="global temperature value")
        parser.add_argument(
            '--matching_global_temperature_fixed', default=False, action="store_true",
            help="True if the global temperature value is constant")
        parser.add_argument(
            '--matching_temperature_weight', type=float, default=1,
            help="True if the temperature weight value is constant")
        parser.add_argument(
            '--matching_temperature_weight_fixed', default=False, action="store_true",
            help="True if the temperature weight value is constant")

        # matching functions and hyper parameters
        parser.add_argument(
            '--feature_projection_dimension', default=1152, type=int,
            help="Dimension of the projection head")
        parser.add_argument(
            "--matching_function", default="otam",
            choices=["mean", "diag", "otam", "linear", "max", "chamfer-query", "chamfer-support",
                     "chamfer"],
            help="matching function")
        parser.add_argument(
            "--video_to_class_matching", default="separate", choices=["separate", "joint"],
            type=str, help="whether to use separate matching or joint matching")
        parser.add_argument(
            "--clip_tuple_length", default=1, type=int, help="length of the clip tuples")
        parser.add_argument(
            "--visil", default=False, action="store_true",
            help="whether to apply the visil fully connected layers before the matching function")

        args = parser.parse_args()

        if args.checkpoint_dir == None:
            print("need to specify a checkpoint dir")
            exit(1)

        if args.backbone == "resnet50":
            # equivalent as trans_linear_in_dim in the TRX repository
            args.backbone_feature_dimension = 2048
        else:
            args.backbone_feature_dimension = 512

        # if the backbone is R2+1D, it will load precomputed features
        args.load_features = args.backbone.startswith("r2+1d")

        if args.dataset_name == "ssv2":
            args.first_val_iter = 10000
            args.val_iter_freq = 10000
            args.training_iterations = 150002
            args.print_freq = 1000
            args.save_freq = 10000
        else:
            args.first_val_iter = 1000
            args.val_iter_freq = 1000
            args.training_iterations = 20002
            args.print_freq = 100
            args.save_freq = 1000

        i_maximum_iter = (args.training_iterations - args.first_val_iter) // args.val_iter_freq + 1
        args.val_iters = [args.first_val_iter + i * args.val_iter_freq for i in range(i_maximum_iter)]

        args.test_only = args.get_best_val_checkpoint or args.test_model_name is not None

        return args

    def run(self):
        train_accuracies = []
        losses = []
        total_iterations = self.args.training_iterations

        iteration = self.start_iteration
        print(f"start iteration {iteration}")

        val_accuraies = [0] * 5
        best_val_accuracy = 0
        timings = []

        for task_dict in self.video_loader:
            if iteration >= total_iterations:
                break

            t_start_iteration = time.time()
            iteration += 1
            torch.set_grad_enabled(True)

            task_loss, task_accuracy = self.train_task(task_dict)
            train_accuracies.append(task_accuracy)
            losses.append(task_loss)

            # optimize
            if ((iteration + 1) % self.args.tasks_per_batch == 0) or (
                    iteration == (total_iterations - 1)):
                if self.args.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm(self.model.parameters(), self.args.gradient_clip)
                self.optimizer.step()
                self.optimizer.zero_grad()

            self.scheduler.step()
            t_end_iteration = time.time()
            timings.append(t_end_iteration - t_start_iteration)

            # print training stats
            if (iteration + 1) % self.args.print_freq == 0 or iteration == 1:
                print_and_log(self.logfile,
                              'Task [{}/{}], Train Loss: {:.7f}, Train Accuracy: {:.7f}'
                              .format(iteration + 1, total_iterations,
                                      torch.Tensor(losses).mean().item(),
                                      torch.Tensor(train_accuracies).mean().item()))
                train_accuracies = []
                losses = []
                timings = []

            # validate
            if (((iteration + 1) in self.args.val_iters) and (
                    iteration + 1) != total_iterations) or iteration == 1:
                accuracy_dict = self.evaluate("val")
                iter_acc = accuracy_dict[self.args.dataset]["accuracy"]
                val_accuraies.append(iter_acc)
                self.val_accuracies.print(self.logfile, accuracy_dict, mode="val")

                # save checkpoint if best validation score
                if iter_acc > best_val_accuracy:
                    best_val_accuracy = iter_acc
                    self.save_checkpoint(iteration + 1, "checkpoint_best_val.pt")

                if self.args.val_on_test:
                    accuracy_dict = self.evaluate("test")
                    self.val_accuracies.print(self.logfile, accuracy_dict, mode="test")

                # # get out if best accuracy was two validations ago
                # if val_accuraies[-1] < val_accuraies[-3]:
                #     break

        # save the final model
        self.save_checkpoint(iteration + 1, "checkpoint_final.pt")

        # evaluate best validation model if it exists, otherwise evaluate the final model.
        try:
            self.load_checkpoint("checkpoint_best_val.pt")
        except:
            self.load_checkpoint("checkpoint_final.pt")

        accuracy_dict = self.evaluate("test")
        print_and_log(self.logfile, f'Best Val Iteration {self.start_iteration}')
        self.val_accuracies.print(self.logfile, accuracy_dict, mode="test")

        self.logfile.close()

    def train_task(self, task_dict):
        """
        For one task, runs forward, calculates the loss and accuracy and backprops
        """
        task_dict = self.prepare_task(task_dict)
        model_dict = self.model(
            task_dict['support_set'], task_dict['support_labels'], task_dict['target_set'])
        target_logits = model_dict['logits']

        task_loss = self.model.loss(task_dict, model_dict) / self.args.tasks_per_batch
        task_accuracy = self.accuracy_fn(target_logits, task_dict['target_labels'])

        task_loss.backward(retain_graph=False)

        return task_loss, task_accuracy

    def compute_accuracies_from_loaded_episodes(self, saved_episodes_dir, n_episodes):
        accuracies = []
        for episode_id in range(1, n_episodes + 1):
            support_set, support_labels, target_set, target_labels = load_episode(
                saved_episodes_dir, episode_id)

            model_dict = self.model(support_set, support_labels, target_set)
            target_logits = model_dict['logits']

            accuracy = self.accuracy_fn(target_logits, target_labels)
            accuracies.append(accuracy.item())
            del target_logits
        return accuracies

    def compute_accuracies_from_dataloader(self, saved_episodes_dir, n_episodes):
        accuracies = []
        iteration = 0
        for task_dict in self.video_loader:
            if iteration >= n_episodes:
                break
            iteration += 1

            task_dict = self.prepare_task(task_dict)
            if self.args.save_test_episodes:
                save_episode(saved_episodes_dir, iteration, task_dict)

            model_dict = self.model(
                task_dict['support_set'], task_dict['support_labels'], task_dict['target_set'])
            target_logits = model_dict['logits']
            accuracy = self.accuracy_fn(target_logits, task_dict['target_labels'])
            accuracies.append(accuracy.item())
            del target_logits
        return accuracies

    def evaluate(self, mode="val"):

        saved_episodes_dir = None
        if self.args.save_test_episodes or self.args.load_test_episodes:
            saved_episodes_dir = get_saved_episode_dir(self.args)

        self.model.eval()
        with torch.no_grad():
            if mode == "val":
                n_tasks = self.args.num_val_tasks
            elif mode == "test":
                n_tasks = self.args.num_test_tasks

            accuracy_dict = {}
            item = self.args.dataset
            if self.args.load_test_episodes:
                accuracies = self.compute_accuracies_from_loaded_episodes(
                    saved_episodes_dir, n_tasks)
            else:
                self.video_loader.dataset.split = mode
                accuracies = self.compute_accuracies_from_dataloader(saved_episodes_dir, n_tasks)
                self.video_loader.dataset.split = "train"

            accuracy = np.array(accuracies).mean() * 100.0
            # 95% confidence interval
            confidence = (196.0 * np.array(accuracies).std()) / np.sqrt(len(accuracies))

            accuracy_dict[item] = {"accuracy": accuracy, "confidence": confidence}

        self.model.train()
        
        return accuracy_dict

    def prepare_task(self, task_dict):
        """
        Remove first batch dimension (as we only ever use a batch size of 1) and move data to device.
        """
        for k in task_dict.keys():
            if k in {"support_video_names", "target_video_names", "support_temporal_positions",
                     "target_temporal_positions"}:
                continue
            task_dict[k] = task_dict[k][0].to(self.device)
        return task_dict

    def save_checkpoint(self, iteration, name="checkpoint.pt"):
        d = {'iteration': iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()}

        torch.save(d, os.path.join(self.checkpoint_dir, 'checkpoint{}.pt'.format(iteration)))   
        torch.save(d, os.path.join(self.checkpoint_dir, name))

    def load_checkpoint(self, name="checkpoint.pt", test_only=False):
        checkpoint_name = os.path.join(self.checkpoint_dir, name)
        print(f"Loading {checkpoint_name}")
        checkpoint = torch.load(checkpoint_name)
        self.start_iteration = checkpoint['iteration']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if not test_only:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])

    def run_test(self, mode):

        if mode == "val":
            n_tasks = self.args.num_val_tasks
        else:
            n_tasks = self.args.num_test_tasks

        if self.args.way != 5:
            suffix = f"_{self.args.way}ways"
        else:
            suffix = ""

        model_name = self.args.test_model_name.split(".")[0]
        logfile_path = os.path.join(
            self.args.checkpoint_dir, f"{model_name}_log_{mode}_{n_tasks}{suffix}.txt")
        logfile_test = open(logfile_path, "a", buffering=1)
        print(f"load {self.args.test_model_name}")
        self.load_checkpoint(self.args.test_model_name, test_only=self.args.test_only)

        self.test_accuracies = TestAccuracies([self.args.dataset])
        t0 = time.time()
        accuracy_dict = self.evaluate(mode=mode)
        t1 = time.time()
        print(f"evaluation lasted {t1 - t0}s")
        logfile_test.write(f"seed {self.args.seed}\n")
        logfile_test.write(f"iteration {self.start_iteration}\n")
        self.test_accuracies.print(logfile_test, accuracy_dict, mode)
        item = self.args.dataset
        self.writer.add_scalar('Accuracy/test', accuracy_dict[item]["accuracy"], self.start_iteration)
        self.writer.add_scalar('Confidence/test', accuracy_dict[item]["confidence"], self.start_iteration)
        # print(f"global_temperature {self.model.global_temperature}")
        # print(f"temperature_weight {self.model.temperature_weight}")
        logfile_test.close()


def main():
    learner = Learner()
    if learner.args.get_best_val_checkpoint:
        print(f"get_best_val_checkpoint")
        learner.args.test_model_name = os.path.join(
            learner.args.checkpoint_dir, f"checkpoint_best_val.pt")
        learner.run_test(learner.args.evaluation_mode)
    elif learner.args.test_model_name is not None:
        print(f"run test")
        learner.args.test_model_name = os.path.join(
            learner.args.checkpoint_dir, learner.args.test_model_name)
        learner.run_test(learner.args.evaluation_mode)
    else:
        print(f"train")
        learner.run()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    main()
