from libai.config import LazyCall
from libai.models import VisionTransformer
from .graph import graph
from .train import train
from .optim import optim
from .imagenet import dataloader

from flowvision.data import Mixup
from flowvision.loss.cross_entropy import SoftTargetCrossEntropy

# Refine data path to imagenet
dataloader.train.dataset[0].root = "/data/dataset/ImageNet/extract"
dataloader.test[0].dataset.root = "/data/dataset/ImageNet/extract"

cfg = dict(
    img_size=224,
    patch_size=16,
    in_chans=3,
    embed_dim=768,
    depth=12,
    num_heads=12,
    mlp_ratio=4.0,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    drop_path_rate=0.0,
    num_classes=1000,
    loss_func=None,
)
model = LazyCall(VisionTransformer)(cfg=cfg)


# Refine model cfg for vit training on imagenet
model.cfg.num_classes = 1000
model.cfg.loss_func = SoftTargetCrossEntropy()

# Add Mixup Func
dataloader.train.mixup_func = LazyCall(Mixup)(
    mixup_alpha=0.8,
    cutmix_alpha=1.0,
    prob=1.0,
    switch_prob=0.5,
    mode="batch",
    num_classes=model.cfg.num_classes,
)

# Refine optimizer cfg for vit model
optim.lr = 1e-3  # 5e-4 * 1024 (batchsize) / 512
optim.eps = 1e-8
optim.weight_decay = 0.05
optim.params.clip_grad_max_norm = None
optim.params.clip_grad_norm_type = None
optim.params.overrides = {"pos_embed": {"weight_decay": 0.0}, "cls_token": {"weight_decay": 0.0}}


# Refine train cfg for vit model
train.train_micro_batch_size = 128
train.test_micro_batch_size = 128
train.train_epoch = 300
train.warmup_ratio = 5 / 300
train.evaluation.eval_period = 1000
train.log_period = 20

# Scheduler
train.scheduler.warmup_factor = 0.001
train.scheduler.alpha = 0.01
train.scheduler.warmup_method = "linear"

# Set fp16 ON
train.amp.enabled = True

# Distributed Settings
train.dist.pipeline_num_layers = model.cfg.depth
train.dist.data_parallel_size = 1
train.dist.tensor_parallel_size = 1
train.dist.pipeline_parallel_size = 1
