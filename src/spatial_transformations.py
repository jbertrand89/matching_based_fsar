from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor)

from mean import get_mean, get_std


def get_spatial_transorm(cmd_args):
    """ Composes the spatial transformations.

    :param cmd_args: command line parameters
    :return: spatial transformations to be applied toeach frame of the clip
    """
    norm_method = get_norm_method(cmd_args)
    if cmd_args.train_crop == "random":
        spatial_transform = get_spatial_transform_with_random_crop_and_flip(cmd_args, norm_method)
    else:
        spatial_transform = get_spatial_transform_center_cropped(cmd_args, norm_method)
    return spatial_transform


def get_spatial_transform_center_cropped(cmd_args, norm_method):
    """
    Composes the spatial transformations with a center cropping only. This method is used for the
    val and test splits.

    :param cmd_args: command line parameters
    :param norm_method: normalization function
    :return: spatial transformations to be applied to each frame of the clip
    """
    return Compose([
        Scale(cmd_args.image_size),
        CenterCrop(cmd_args.image_size),
        ToTensor(cmd_args.norm_value), norm_method
    ])


def get_spatial_transform_with_random_crop_and_flip(cmd_args, norm_method):
    """
    Composes the spatial transformations including random cropping and horizontal flipping (if
    spatial_transform_horizontal_flip is set to true). This method is used for the train split.

    :param cmd_args: command line parameters
    :param norm_method: normalization function
    :return: spatial transformations to be applied to each frame of the clip
    """
    scales = get_scales(cmd_args)
    crop_method = get_crop_method(cmd_args, scales)

    transforms = [crop_method]

    if cmd_args.spatial_transform_horizontal_flip:
        transforms.append(RandomHorizontalFlip())

    transforms.append(ToTensor(cmd_args.norm_value))
    transforms.append(norm_method)

    return Compose(transforms)


def get_norm_method(cmd_args):
    """
    Defines the normalization function as defined in the few-shot-video-classification repository.
    It requires the config parameters norm_value to be defined.

    :param cmd_args: command line parameters
    :return: normalization function
    """
    mean = get_mean(cmd_args.norm_value)
    std = get_std(cmd_args.norm_value)

    if cmd_args.no_mean_norm and not cmd_args.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not cmd_args.std_norm:
        norm_method = Normalize(mean, [1, 1, 1])
    else:
        norm_method = Normalize(mean, std)
    return norm_method


def get_crop_method(cmd_args, scales):
    """
    Defines the crop function to be either random, corner or center. It requires the config
    parameters image_size to be defined.

    :param cmd_args: command line parameters
    :param scales: image scaling factors
    :return: cropping function
    """
    if cmd_args.train_crop == 'random':
        crop_method = MultiScaleRandomCrop(scales, cmd_args.image_size)
    elif cmd_args.train_crop == 'corner':
        crop_method = MultiScaleCornerCrop(scales, cmd_args.image_size)
    elif cmd_args.train_crop == 'center':
        crop_method = MultiScaleCornerCrop(
            scales, cmd_args.image_size, crop_positions=['c'])
    return crop_method


def get_scales(cmd_args):
    """Defines the scales used for cropping. It requires the config parameters initial_scale and
    scale_step to be defined

    :param cmd_args: command line parameters
    :return: image scaling factors
    """
    scales = [cmd_args.initial_scale]
    for _ in range(1, cmd_args.n_scales):
        scales.append(scales[-1] * cmd_args.scale_step)
    return scales

