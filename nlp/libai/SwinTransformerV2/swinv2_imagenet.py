from libai.config import LazyCall
from .graph import graph
from .train import train
from .optim import optim
from .imagenet import dataloader

from flowvision import transforms
from flowvision.data import Mixup
from flowvision.loss.cross_entropy import SoftTargetCrossEntropy
from flowvision.transforms import InterpolationMode
from flowvision.transforms.functional import str_to_interp_mode
from flowvision.data.constants import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
)
from flowvision.data.auto_augment import rand_augment_transform
from flowvision.data.random_erasing import RandomErasing

from omegaconf import DictConfig
from libai.models import SwinTransformerV2

cfg = dict(
    img_size=256,
    patch_size=4,
    in_chans=3,
    num_classes=1000,
    embed_dim=96,
    depths=[2, 2, 6, 2],
    num_heads=[3, 6, 12, 24],
    window_size=8,
    mlp_ratio=4.0,
    qkv_bias=True,
    drop_rate=0.0,
    drop_path_rate=0.2,
    ape=False,
    patch_norm=True,
    pretrained_window_sizes=[0, 0, 0, 0],
    loss_func=None,
)

cfg = DictConfig(cfg)

model = LazyCall(SwinTransformerV2)(cfg=cfg)

# Refine data path to imagenet
dataloader.train.dataset[0].root = "/data/dataset/ImageNet/extract"
dataloader.test[0].dataset.root = "/data/dataset/ImageNet/extract"

# Add Mixup Func
dataloader.train.mixup_func = LazyCall(Mixup)(
    mixup_alpha=0.8,
    cutmix_alpha=1.0,
    prob=1.0,
    switch_prob=0.5,
    mode="batch",
    num_classes=1000,
)

dataloader.train.dataset[0].transform = LazyCall(transforms.Compose)(
    transforms=[
        LazyCall(transforms.RandomResizedCrop)(
            size=256,
            scale=(0.08, 1.0),
            ratio=(3.0 / 4.0, 4.0 / 3.0),
            interpolation=InterpolationMode.BICUBIC,
        ),
        LazyCall(transforms.RandomHorizontalFlip)(p=0.5),
        LazyCall(rand_augment_transform)(
            config_str="rand-m9-mstd0.5-inc1",
            hparams=dict(
                translate_const=int(256 * 0.45),
                img_mean=tuple([min(255, round(255 * x)) for x in IMAGENET_DEFAULT_MEAN]),
                interpolation=str_to_interp_mode("bicubic"),
            ),
        ),
        LazyCall(transforms.ToTensor)(),
        LazyCall(transforms.Normalize)(
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        ),
        LazyCall(RandomErasing)(
            probability=0.25,
            mode="pixel",
            max_count=1,
            num_splits=0,
            device="cpu",
        ),
    ]
)
dataloader.test[0].dataset.transform = LazyCall(transforms.Compose)(
    transforms=[
        LazyCall(transforms.Resize)(
            size=256,
            interpolation=InterpolationMode.BICUBIC,
        ),
        LazyCall(transforms.CenterCrop)(
            size=256,
        ),
        LazyCall(transforms.ToTensor)(),
        LazyCall(transforms.Normalize)(
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        ),
    ]
)


# Refine model cfg for vit training on imagenet
model.cfg.num_classes = 1000
model.cfg.loss_func = SoftTargetCrossEntropy()

# Refine optimizer cfg for vit model
optim.lr = 1e-3  # The pytorch version is 1024 as the total batch size, 1e-3 as the learning rate
optim.eps = 1e-8
optim.weight_decay = 0.05


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin


def set_weight_decay(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if (
            len(param.shape) == 1
            or name.endswith(".bias")
            or (name in skip_list)
            or check_keywords_in_name(name, skip_keywords)
        ):
            no_decay.append(param)
        else:
            has_decay.append(param)
    return [{"params": has_decay}, {"params": no_decay, "weight_decay": 0.0}]


optim.params = LazyCall(set_weight_decay)(
    model=model,
    skip_list=("absolute_pos_embed"),
    skip_keywords=("cpb_mlp", "logit_scale", "relative_position_bias_table"),
)

# Refine train cfg for vit model
train.train_micro_batch_size = 64
train.test_micro_batch_size = 128
train.train_epoch = 300
train.warmup_ratio = 20 / 300
train.eval_period = 1562
train.log_period = 100
graph.enabled = False
train.rdma_enabled = True
# Scheduler
train.scheduler.warmup_factor = 0.001
train.scheduler.alpha = 0.01
train.scheduler.warmup_method = "linear"
# Set fp16 ON
train.amp.enabled = True
