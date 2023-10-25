"""
Flowvision training scheduler by flowvision contributors
"""

from typing import List

from oneflow.optim.lr_scheduler import LinearLR
from oneflow.optim.lr_scheduler import StepLR
from oneflow.optim.lr_scheduler import MultiStepLR
from oneflow.optim.lr_scheduler import CosineAnnealingLR

def build_scheduler(config, optimizer, n_iter_per_epoch):
    num_steps = int(config.TRAIN.EPOCHS * n_iter_per_epoch)
    warmup_steps = int(config.TRAIN.WARMUP_EPOCHS * n_iter_per_epoch)
    if config.TRAIN.LR_SCHEDULER.NAME == "multi_step":
        assert (
            isinstance(config.TRAIN.LR_SCHEDULER.MILESTONES, List)
        ), "decay_t must be a list of epoch indices which are increasing only when you're using multi-step lr scheduler."
        decay_steps = [
            decay_step * n_iter_per_epoch
            for decay_step in config.TRAIN.LR_SCHEDULER.MILESTONES
        ]
    else:
        decay_steps = int(config.TRAIN.LR_SCHEDULER.DECAY_EPOCHS * n_iter_per_epoch)

    lr_scheduler = None
    if config.TRAIN.LR_SCHEDULER.NAME == "cosine":
        lr_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_steps,
        )
    elif config.TRAIN.LR_SCHEDULER.NAME == "linear":
        lr_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,
            total_iters=num_steps,
        )
    elif config.TRAIN.LR_SCHEDULER.NAME == "step":
        lr_scheduler = StepLR(
            optimizer,
            step_size=decay_steps,
            gamma=config.TRAIN.LR_SCHEDULER.DECAY_RATE,
        )
    elif config.TRAIN.LR_SCHEDULER.NAME == "multi_step":
        lr_scheduler = MultiStepLR(
            optimizer,
            milestones=decay_steps,
            gamma=config.TRAIN.LR_SCHEDULER.DECAY_RATE,
        )

    return lr_scheduler
