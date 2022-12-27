from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor)

from mean import get_mean, get_std


def get_spatial_transorm(opt):
    """
    Composes the spatial transformations.
    """
    norm_method = get_norm_method(opt)
    if opt.train_crop == "random":
        spatial_transform = get_spatial_transform_with_random_crop_and_flip(opt, norm_method)
    else:
        spatial_transform = get_spatial_transform_center_cropped(opt, norm_method)
    return spatial_transform


def get_spatial_transform_center_cropped(opt, norm_method):
    """
    Composes the spatial transformations with a center cropping only. This method is used for the
    val and test splits.
    """
    return Compose([
        Scale(opt.image_size),
        CenterCrop(opt.image_size),
        ToTensor(opt.norm_value), norm_method
    ])


def get_spatial_transform_with_random_crop_and_flip(opt, norm_method):
    """
    Composes the spatial transformations including random cropping and horizontal flipping (if
    spatial_transform_horizontal_flip is set to true). This method is used for the train split.
    """
    scales = get_scales(opt)
    crop_method = get_crop_method(opt, scales)

    transforms = [crop_method]

    if opt.spatial_transform_horizontal_flip:
        transforms.append(RandomHorizontalFlip())

    transforms.append(ToTensor(opt.norm_value))
    transforms.append(norm_method)

    return Compose(transforms)


def get_norm_method(opt):
    """
    Defines the normalization function as defined in the few-shot-video-classification repository.
    It requires the config parameters norm_value to be defined.
    """
    mean = get_mean(opt.norm_value)
    std = get_std(opt.norm_value)

    if opt.no_mean_norm and not opt.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        norm_method = Normalize(mean, [1, 1, 1])
    else:
        norm_method = Normalize(mean, std)
    return norm_method


def get_crop_method(opt, scales):
    """
    Defines the crop function to be either random, corner or center. It requires the config
    parameters image_size to be defined.
    """
    if opt.train_crop == 'random':
        crop_method = MultiScaleRandomCrop(scales, opt.image_size)
    elif opt.train_crop == 'corner':
        crop_method = MultiScaleCornerCrop(scales, opt.image_size)
    elif opt.train_crop == 'center':
        crop_method = MultiScaleCornerCrop(
            scales, opt.image_size, crop_positions=['c'])
    return crop_method


def get_scales(opt):
    """Defines the scales used for cropping. It requires the config parameters initial_scale and
    scale_step to be defined"""
    scales = [opt.initial_scale]
    for _ in range(1, opt.n_scales):
        scales.append(scales[-1] * opt.scale_step)
    return scales

