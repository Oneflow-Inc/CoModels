"""
Modified from https://github.com/microsoft/Swin-Transformer/blob/main/data/build.py
"""

import os
import numpy as np

import oneflow as flow
from oneflow.utils.data import DataLoader

from flowvision import datasets, transforms
from flowvision.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from flowvision.data import create_transform
from flowvision.transforms.functional import str_to_interp_mode
from flowvision.data import Mixup

from .cached_image_folder import CachedImageFolder
from .samplers import SubsetRandomSampler


IMAGENET_HEIGHT = 224
IMAGENET_WIDTH = 224
IMAGENET_NUM_CLASSES = 1000


def build_loader(config):
    config.defrost()
    if config.DATA.SYNTHETIC_DATA == True:
        print("use synthetic data")
        dataset_train = None
        dataset_val = None
        data_loader_train = SyntheticDataLoader(batch_size=config.DATA.BATCH_SIZE, length=200)
        data_loader_val = SyntheticDataLoader(batch_size=config.DATA.BATCH_SIZE, length=100)
    else:
        dataset_train, config.MODEL.NUM_CLASSES = build_dataset(
            is_train=True, config=config
        )
        config.freeze()
        print(
            f"local rank {config.LOCAL_RANK} / global rank {flow.env.get_rank()} successfully build train dataset"
        )
        dataset_val, _ = build_dataset(is_train=False, config=config)
        print(
            f"local rank {config.LOCAL_RANK} / global rank {flow.env.get_rank()} successfully build val dataset"
        )

        num_tasks = flow.env.get_world_size()
        global_rank = flow.env.get_rank()
        if config.DATA.ZIP_MODE and config.DATA.CACHE_MODE == "part":
            indices = np.arange(
                flow.env.get_rank(), len(dataset_train), flow.env.get_world_size()
            )
            sampler_train = SubsetRandomSampler(indices)
        else:
            sampler_train = flow.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )

        indices = np.arange(
            flow.env.get_rank(), len(dataset_val), flow.env.get_world_size()
        )
        sampler_val = SubsetRandomSampler(indices)
        data_loader_train = DataLoader(
            dataset_train,
            sampler=sampler_train,
            batch_size=config.DATA.BATCH_SIZE,
            num_workers=config.DATA.NUM_WORKERS,
            persistent_workers=True,
            drop_last=True,
        )

        data_loader_val = DataLoader(
            dataset_val,
            sampler=sampler_val,
            batch_size=config.DATA.BATCH_SIZE,
            shuffle=False,
            num_workers=config.DATA.NUM_WORKERS,
            persistent_workers=True,
            drop_last=False,
        )

    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = (
        config.AUG.MIXUP > 0
        or config.AUG.CUTMIX > 0.0
        or config.AUG.CUTMIX_MINMAX is not None
    )
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP,
            cutmix_alpha=config.AUG.CUTMIX,
            cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB,
            switch_prob=config.AUG.MIXUP_SWITCH_PROB,
            mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING,
            num_classes=config.MODEL.NUM_CLASSES,
        )

    return dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn


def build_dataset(is_train, config):
    transform = build_transform(is_train, config)
    if config.DATA.DATASET == "imagenet":
        prefix = "train" if is_train else "val"
        if config.DATA.ZIP_MODE:
            ann_file = prefix + "_map.txt"
            prefix = prefix + ".zip@/"
            dataset = CachedImageFolder(
                config.DATA.DATA_PATH,
                ann_file,
                prefix,
                transform,
                cache_mode=config.DATA.CACHE_MODE if is_train else "part",
            )
        else:
            root = os.path.join(config.DATA.DATA_PATH, prefix)
            dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif config.DATA.DATASET == "cifar100":
        dataset = datasets.CIFAR100(
            root=config.DATA.DATA_PATH,
            train=is_train,
            transform=transform,
            download=False,
        )
        nb_classes = 100
    else:
        raise NotImplementedError("We only support ImageNet and CIFAR100 Now.")

    return dataset, nb_classes


def build_transform(is_train, config):
    resize_im = config.DATA.IMG_SIZE > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=config.DATA.IMG_SIZE,
            is_training=True,
            color_jitter=config.AUG.COLOR_JITTER
            if config.AUG.COLOR_JITTER > 0
            else None,
            auto_augment=config.AUG.AUTO_AUGMENT
            if config.AUG.AUTO_AUGMENT != "none"
            else None,
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            interpolation=config.DATA.INTERPOLATION,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                config.DATA.IMG_SIZE, padding=4
            )
        return transform

    t = []
    if resize_im:
        if config.TEST.CROP:
            size = int((256 / 224) * config.DATA.IMG_SIZE)
            t.append(
                transforms.Resize(
                    size, interpolation=str_to_interp_mode(config.DATA.INTERPOLATION)
                ),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
        else:
            t.append(
                transforms.Resize(
                    (config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                    interpolation=str_to_interp_mode(config.DATA.INTERPOLATION),
                )
            )

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


class SyntheticDataLoader(object):
    def __init__(
        self,
        batch_size,
        image_size=224,
        num_classes=1000,
        channel_last=False,
        length=1000,
    ):
        super().__init__()

        if channel_last:
            self.image_shape = (batch_size, image_size, image_size, 3)
        else:
            self.image_shape = (batch_size, 3, image_size, image_size)
        self.label_shape = (batch_size,)
        self.num_classes = num_classes

        self.image = flow.randint(
            0, high=256, size=self.image_shape, dtype=flow.float32, device="cpu"
        )
        self.label = flow.randint(
            0, high=self.num_classes, size=self.label_shape, device="cpu",
        ).to(dtype=flow.int64)

        self.count = 0
        self.length = length

    def _reset(self):
        self.count = 0
    
    def __next__(self):
        if self.count < self.length:
            self.count += 1
            return self.image, self.label
        else:
            self._reset()
            raise StopIteration
    
    def __iter__(self):
        return self
    

    def __len__(self):
        return self.length
