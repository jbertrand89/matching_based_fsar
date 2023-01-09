import math
import numpy as np
import os
import torch


class VideoFeatureReader(torch.utils.data.Dataset):
    """ Creates video episodes from pre-saved video features.
    It loads the episode from a dataset split defined by:
      - a split name (train/val/test)
      - a split directory, where the video features are presaved for this split
      - a split seed, to use different random generators with fixed seed for each split. It enables
      to enforce that the validation episodes are identical from each training iteration
    """
    def __init__(self, args, split_name, split_dir, split_seed):
        self.device = args.device
        self.split_name = split_name

        # loads all the feature filenames
        self.filenames_by_class = self.load_dataset_filenames(split_dir)

        # creates a generator per split and associated a fixed seed to it
        self.seed = int(split_seed)
        self.reset_generator()

        self.clip_count = args.seq_len
        self.class_count = args.way  # also named C way in the literature
        self.example_per_class_count = args.shot  # also named k shot
        self.query_per_class = args.query_per_class
        self.query_per_class_test = args.query_per_class_test

    def __len__(self):
        return 1000000000

    def load_dataset_filenames(self, split_dir):
        """ Loads the filenames of the split dataset. The split directory follows this
        format:
        ├── split_dir
        │   ├── class_name_0
        │   │   ├── video_name_0
        │   │   │   ── video_feature.pth
        │   │   ├── video_name_1
        │   │   │   ── video_feature.pth
        │   │   └── ...
        │   └── class_name_1
        │   │   ├── video_name_0
        │   │   │   ── video_feature.pth
        │   │   ├── video_name_1
        │   │   │   ── video_feature.pth
        │   │   └── ...
        │   └── ...

        :param split_dir: directory containing the split features
        :return: dictionary containing the list of feature filenames for each class in the given
          split directory
        """
        filenames_by_class = {}
        count = 0
        for class_name in os.listdir(split_dir):
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            filenames_by_class[class_name] = [os.path.join(class_dir, p) for p in os.listdir(class_dir)]
            count += len(os.listdir(class_dir))
        print(f"Loaded {split_dir} count {count}")
        return filenames_by_class

    def reset_generator(self):
        """ Resets the random generator. It is useful for enforcing the dataloader to see the exact
        same validation examples for each training iteration.
        """
        self.random_generator = np.random.default_rng(self.seed)

    def load_video_features(self, video_dir, random_generator):
        """ Loads all the features for a video and selects n (self.clip_count) features uniformly
        sampled.

        :param video_dir: directory containing all the features for the current video
        :param random_generator: random generator with fixed seed
        :return: n clip features
        :return: n clip temporal positions
        """
        # for each video, load all the clip features
        filename = os.path.join(video_dir, "video_features.pth")
        video_features = torch.load(filename, map_location=self.device)

        # when there is only one feature (video of 16 frames), reshape the tensor
        if len(video_features.shape) == 1 and video_features.shape[0] == 512:
            video_features = video_features.reshape(1, -1)

        # if the video is too small to extract n clips, repeat the last clip
        feature_count = video_features.shape[0]
        if feature_count <= self.clip_count:
            while video_features.shape[0] < self.clip_count:
                video_features = torch.cat([video_features, video_features[-1:]], dim=0)
            return video_features, [0]

        # do temporal filter selection (with fixed seed)
        boundaries = np.linspace(0, feature_count, self.clip_count + 1)
        temporal_positions = [
            random_generator.choice(range(math.ceil(boundaries[i]), math.ceil(boundaries[i + 1])))
            for i in range(self.clip_count)]

        video_features = video_features[temporal_positions]

        return video_features, temporal_positions

    def __getitem__(self, index: int):
        """
        Loads a video episode. It is composed of:
          - the support video features from class_count classes (usually 5) and
          example_per_class_count video examples per class (usually between 1 to 5)
          - the support video labels for each support examples
          - the target/query video features from the same classes as the support but different
          video examples
          - the target/query video labels for each support examples

        :param index: index
        :return: dictionary containing the support features, the support labels, the target features
          and the target labels
        """
        dataset = self.filenames_by_class

        if self.split_name == "train":
            n_queries = self.query_per_class
        else:
            n_queries = self.query_per_class_test
        way = self.class_count

        # pick c classes (with seed)
        all_classes = np.array(list(dataset.keys()))
        class_count = len(dataset.keys())
        classes_idx = self.random_generator.permutation(class_count)[0: way]
        batch_classes = all_classes[classes_idx]

        support_set = []
        support_labels = []
        target_set = []
        target_labels = []
        for label_id, class_id in enumerate(batch_classes):
            # pick k + q elements names (with seed)
            video_count = len(dataset[class_id])
            example_count = self.example_per_class_count + n_queries
            class_video_idx = self.random_generator.permutation(video_count)[0: example_count]
            class_video_names = np.array(dataset[class_id])[class_video_idx]
            class_support_video_names = class_video_names[0: self.example_per_class_count]
            class_query_video_names = class_video_names[self.example_per_class_count:]

            for video_dir in class_support_video_names:
                video_features, temporal_positions = self.load_video_features(
                    video_dir, self.random_generator)
                support_set.append(video_features)
                support_labels.append(label_id)

            for video_dir in class_query_video_names:
                video_features, temporal_positions = self.load_video_features(
                    video_dir, self.random_generator)
                target_set.append(video_features)
                target_labels.append(label_id)

        support_set = torch.stack(support_set, dim=0)
        target_set = torch.stack(target_set, dim=0)
        support_labels = torch.FloatTensor(support_labels)
        target_labels = torch.FloatTensor(target_labels)

        return {"support_set": support_set, "support_labels": support_labels,
                "target_set": target_set, "target_labels": target_labels}
