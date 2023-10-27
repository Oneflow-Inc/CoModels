from libai.config import LazyCall
from omegaconf import DictConfig
from libai.models import SwinTransformer
from .graph import graph
from .train import train
from .optim import optim
from .imagenet import dataloader

from flowvision.data import Mixup
from flowvision.loss.cross_entropy import SoftTargetCrossEntropy

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

cfg = dict(
    img_size=224,
    patch_size=4,
    in_chans=3,
    num_classes=1000,
    embed_dim=96,
    depths=[2, 2, 6, 2],
    num_heads=[3, 6, 12, 24],
    window_size=7,
    mlp_ratio=4.0,
    qkv_bias=True,
    qk_scale=None,
    drop_rate=0.0,
    drop_path_rate=0.2,
    ape=False,
    patch_norm=True,
    loss_func=None,
)

cfg = DictConfig(cfg)

model = LazyCall(SwinTransformer)(cfg=cfg)


# Refine model cfg for vit training on imagenet
model.cfg.num_classes = 1000
model.cfg.loss_func = SoftTargetCrossEntropy()
# Refine optimizer cfg for vit model
optim.lr = 1e-3
optim.eps = 1e-8
optim.weight_decay = 0.05
optim.params.clip_grad_max_norm = None
optim.params.clip_grad_norm_type = None

# Refine train cfg for vit model
train.train_micro_batch_size = 128
train.test_micro_batch_size = 128
train.train_epoch = 300
train.warmup_ratio = 20 / 300
train.eval_period = 1562
train.log_period = 100

# Scheduler
train.scheduler.warmup_factor = 0.001
train.scheduler.alpha = 0.01
train.scheduler.warmup_method = "linear"

# Set fp16 ON
train.amp.enabled = True
