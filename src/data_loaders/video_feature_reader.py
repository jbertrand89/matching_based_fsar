import itertools
import math
import numpy as np
import os
import time
import torch


class VideoFeatureReader(torch.utils.data.Dataset):
    def __init__(self, args):
        self.device = args.device
        self.dataset_splits = {}
        self.random_generator = {}
        self.generator_seed = {}
        self.video_id_by_video_name = {}
        self.class_by_video_name = {}
        for split_name, split_path, split_seed in zip(
                args.split_names, args.split_paths, args.split_seeds):
            t0 = time.time()
            dataset = self.load_dataset(split_path)
            t1 = time.time()
            print(f"Loaded dataset for split = {split_name} {split_path} in {t1 - t0}s")
            self.dataset_splits[split_name] = dataset

            filename = os.path.join(split_path, f"{split_name}.txt")
            self.save_dataset_map(filename, dataset)
            self.load_dataset_map(filename, split_name)

            self.generator_seed[split_name] = int(split_seed)
            self.reset_generator(split_name)

        self.split = "train"
        self.clip_count = args.seq_len
        self.way = args.way
        # self.train_way = args.way
        # self.validation_way = args.validation_way
        self.shot = args.shot
        self.query_per_class = args.query_per_class
        self.query_per_class_test = args.query_per_class_test

    def __len__(self):
        return 1000000000

    def get_split(self):
        """ return the current split being used """
        return self.dataset_splits[self.split]

    def load_dataset(self, split_path):
        dataset = {}
        count = 0
        for class_name in os.listdir(split_path):
            if class_name.endswith(".txt"):
                continue
            class_path = os.path.join(split_path, class_name)
            dataset[class_name] = [os.path.join(class_path, p) for p in os.listdir(class_path)]
            count += len(os.listdir(class_path))
        print(f"Loaded {split_path} count {count}")
        return dataset

    def save_dataset_map(self, filename, dataset):
        if not os.path.exists(filename):
            video_id = 0
            with open(filename, "w+") as writer:
                for class_name in dataset:
                    for video_path in dataset[class_name]:
                        video_name = video_path.split("/")[-1]
                        writer.write(";".join([video_name, str(video_id), class_name]) + "\n")
                        video_id += 1
            print(f"Saved {filename}")

    def load_dataset_map(self, filename, split_name):
        print(f"Loading {filename}")
        with open(filename, "r+") as reader:
            data = reader.readlines()

        video_id_by_video_name = {}
        class_by_video_name = {}
        for line in data:
            video_name, video_id, class_name = line.strip().split(";")
            video_id_by_video_name[video_name] = video_id
            class_by_video_name[video_name] = class_name
        self.video_id_by_video_name[split_name] = video_id_by_video_name
        self.class_by_video_name[split_name] = class_by_video_name

    def reset_generator(self, split_name):
        # if self.use_specific_generator:
        self.random_generator[split_name] = np.random.default_rng(self.generator_seed[split_name])

    def load_features(self, video_path, random_generator):
        # for each element, load full features
        filename = os.path.join(video_path, "video_features.pth")
        video_features = torch.load(filename, map_location=self.device)

        # Fix when only one feature
        if len(video_features.shape) == 1 and video_features.shape[0] == 512:
            video_features = video_features.reshape(1, -1)

        # if the video is too small, repeat the last clip
        feature_count = video_features.shape[0]
        if feature_count <= self.clip_count:
            while video_features.shape[0] < self.clip_count:
                video_features = torch.cat([video_features, video_features[-1:]], dim=0)
            return video_features, [0]

        # do temporal filter selection (with seed)
        boundaries = np.linspace(0, feature_count, self.clip_count + 1)
        # temporal_positions = [
        #     np.random.choice(range(math.ceil(boundaries[i]), math.ceil(boundaries[i + 1])))
        #     for i in range(self.snippet_count)]
        temporal_positions = [
            random_generator.choice(range(math.ceil(boundaries[i]), math.ceil(boundaries[i + 1])))
            for i in range(self.clip_count)]

        video_features = video_features[temporal_positions]

        return video_features, temporal_positions

    def get_video_description(self, video_path):
        path_parts = video_path.split("/")
        video_name = path_parts[-1]
        video_id = self.video_id_by_video_name[self.split][video_name]
        class_name = self.class_by_video_name[self.split][video_name]
        return f"video={video_id}/class={class_name}"

    def __getitem__(self, index):
        dataset = self.get_split()
        random_generator = self.random_generator[self.split]

        if self.split == "train":
            n_queries = self.query_per_class
            # way = self.train_way
        else:
            n_queries = self.query_per_class_test
            # way = self.validation_way
        way = self.way

        # pick c classes (with seed)
        all_classes = np.array(list(dataset.keys()))
        class_count = len(dataset.keys())
        classes_idx = random_generator.permutation(class_count)[0: way]
        batch_classes = all_classes[classes_idx]

        support_set = []
        support_labels = []
        target_set = []
        target_labels = []
        support_video_names = []
        target_video_names = []
        real_target_labels = []
        support_temporal_positions = []
        target_temporal_positions = []
        for label_id, class_id in enumerate(batch_classes):
            # pick k + q elements names (with seed)
            video_count = len(dataset[class_id])
            class_video_idx = random_generator.permutation(video_count)[0: self.shot + n_queries]
            class_video_names = np.array(dataset[class_id])[class_video_idx]
            class_support_video_names = class_video_names[0: self.shot]
            class_query_video_names = class_video_names[self.shot:]

            for video_path in class_support_video_names:
                video_features, temporal_positions = self.load_features(
                    video_path, random_generator)
                support_set.append(video_features)
                support_labels.append(label_id)
                video_name = self.get_video_description(video_path)
                support_video_names.append(video_name)
                support_temporal_positions.append("_".join([str(t) for t in temporal_positions]))

            for video_path in class_query_video_names:
                video_features, temporal_positions = self.load_features(
                    video_path, random_generator)

                target_set.append(video_features)
                target_labels.append(label_id)
                target_temporal_positions.append("_".join([str(t) for t in temporal_positions]))

                video_name = self.get_video_description(video_path)
                target_video_names.append(video_name)
                real_target_labels.append(batch_classes[label_id])

        support_set = torch.stack(support_set, dim=0)
        target_set = torch.stack(target_set, dim=0)
        support_labels = torch.FloatTensor(support_labels)
        target_labels = torch.FloatTensor(target_labels)

        return {"support_set": support_set, "support_labels": support_labels,
                "target_set": target_set, "target_labels": target_labels,
                "support_video_names": support_video_names,
                "target_video_names": target_video_names,
                "support_temporal_positions": support_temporal_positions,
                "target_temporal_positions": target_temporal_positions
                }
