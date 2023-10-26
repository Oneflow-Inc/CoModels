"""
Modified from https://github.com/microsoft/Swin-Transformer/blob/main/utils.py
"""

import os
import oneflow as flow
from flowvision.models.utils import load_state_dict_from_url
import time


def load_checkpoint(config, model, optimizer, lr_scheduler, logger):
    logger.info(
        f"==============> Resuming form {config.MODEL.RESUME}...................."
    )
    if config.MODEL.RESUME.startswith("https"):
        checkpoint = load_state_dict_from_url(
            config.MODEL.RESUME, map_location="cpu", check_hash=True
        )
    else:
        checkpoint = flow.load(config.MODEL.RESUME)
    msg = model.load_state_dict(checkpoint["model"], strict=False)
    logger.info(msg)
    max_accuracy = 0.0
    if (
        not config.EVAL_MODE
        and "optimizer" in checkpoint
        and "lr_scheduler" in checkpoint
        and "epoch" in checkpoint
    ):
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint["epoch"] + 1
        config.freeze()
        logger.info(
            f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})"
        )
        if "max_accuracy" in checkpoint:
            max_accuracy = checkpoint["max_accuracy"]

    return max_accuracy


def save_checkpoint(
    config, epoch, model, max_accuracy, optimizer, lr_scheduler, logger
):
    save_state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "max_accuracy": max_accuracy,
        "epoch": epoch,
        "config": config,
    }

    save_path = os.path.join(config.OUTPUT, f"model_{epoch}")
    logger.info(f"{save_path} saving......")
    flow.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")


def get_grad_norm(parameters, norm_type=2):
    total_norm = flow.linalg.vector_norm(
        flow.stack(
            [flow.linalg.vector_norm(p.grad.detach(), norm_type) for p in parameters]
        ),
        norm_type,
    )
    return total_norm


def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [
        ckpt for ckpt in checkpoints if os.path.isdir(os.path.join(output_dir, ckpt))
    ]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max(
            [os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime
        )
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def reduce_tensor(tensor):
    rt = tensor.clone()
    flow.comm.all_reduce(rt)
    rt /= flow.env.get_world_size()
    return rt

class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def record(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def get(self):
        if isinstance(self.val, flow.Tensor):
            return self.val.item(), self.avg.item()
        else:
            return self.val, self.avg


class TimeMeter(object):
    def __init__(self, return_timestamp=False):
        self.return_timestamp = return_timestamp
        self.total_n = 0
        self.n = 0
        self.start_time = None
        self.ets = None
        self.bts = None
        self.reset()

    def reset(self):
        self.n = 0
        if self.start_time is None:
            self.start_time = time.perf_counter()
        if self.ets is None:
            self.bts = time.perf_counter()
        else:
            self.bts = self.ets
        self.ets = None

    def record(self, n):
        self.n += n
        self.total_n += n

    def get(self):
        self.ets = time.perf_counter()
        assert self.ets > self.bts, f"{self.ets} > {self.bts}"
        assert self.ets > self.start_time, f"{self.ets} > {self.start_time}"
        throughput = self.n / (self.ets - self.bts)
        avg_throughput = self.total_n / (self.ets - self.start_time)
        if self.return_timestamp:
            return throughput, avg_throughput, self.ets
        else:
            return throughput, avg_throughput
