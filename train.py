import datetime
import os
import time
import warnings

import torch
import torch.ao.quantization
import torch.utils.data
from torch import nn

from utils import (
    SmoothedValue, 
    MetricLogger, 
    reduce_across_processes, 
    accuracy, 
    mkdir, 
    save_on_master, 
    init_distributed_mode
)
from data_loader import create_dataloader
from models import ExponentialMovingAverage, create_model
from loss_fn import create_optimizer
from resnetv2 import bit_resnet152x2
import numpy as np

def train_one_epoch(
    model, 
    criterion, 
    optimizer, 
    data_loader, 
    device, 
    epoch, 
    args, 
    teacher_model=None, 
    model_ema=None, 
    scaler=None
):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value}"))

    header = f"Epoch: [{epoch}]"
    for i, (image, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        image, target = image.to(device), target.to(device)
        
        if teacher_model is not None:
            teacher_model.eval()
            with torch.no_grad():
                target = teacher_model(image)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target)
            

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            if args.clip_grad_norm is not None:
                # we should unscale the gradients of optimizer's assigned params if do gradient clipping
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()

        if model_ema and i % args.model_ema_steps == 0:
            model_ema.update_parameters(model)
            if epoch < args.lr_warmup_epochs:
                # Reset ema buffer to keep copying weights during warmup period
                model_ema.n_averaged.fill_(0)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)


def evaluate(model, criterion, data_loader, device, print_freq=100, log_suffix=""):
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"

    num_processed_samples = 0
    with torch.inference_mode():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            num_processed_samples += batch_size
    # gather the stats from all processes

    num_processed_samples = reduce_across_processes(num_processed_samples)
    if (
        hasattr(data_loader.dataset, "__len__")
        and len(data_loader.dataset) != num_processed_samples
        and torch.distributed.get_rank() == 0
    ):
        # See FIXME above
        warnings.warn(
            f"It looks like the dataset has {len(data_loader.dataset)} samples, but {num_processed_samples} "
            "samples were used for the validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )

    metric_logger.synchronize_between_processes()

    print(f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}")
    return metric_logger.acc1.global_avg


def main(args):
    if args.output_dir:
        mkdir(args.output_dir)

    init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    num_classes, data_loader, data_loader_test, train_sampler = create_dataloader(args)
    model, parameters = create_model(args, device, num_classes)

    # TODO write a function to modify output layer
    teacher_model = bit_resnet152x2()
    teacher_model.load_from(np.load('BiT-S-R152x2.npz'))
    teacher_model = teacher_model.to(device)
    
    lr_scheduler, scaler, optimizer = create_optimizer(args, parameters)
    #criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    criterion = nn.KLDivLoss()

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    model_ema = None
    if args.model_ema:
        # Decay adjustment that aims to keep the decay independent of other hyper-parameters originally proposed at:
        # https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
        #
        # total_ema_updates = (Dataset_size / n_GPUs) * epochs / (batch_size_per_gpu * EMA_steps)
        # We consider constant = Dataset_size for a given dataset/setup and omit it. Thus:
        # adjust = 1 / total_ema_updates ~= n_GPUs * batch_size_per_gpu * EMA_steps / epochs
        adjust = args.world_size * args.batch_size * args.model_ema_steps / args.epochs
        alpha = 1.0 - args.model_ema_decay
        alpha = min(1.0, alpha * adjust)
        model_ema = ExponentialMovingAverage(model_without_ddp, device=device, decay=1.0 - alpha)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        model_without_ddp.load_state_dict(checkpoint["model"])
        if not args.test_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if model_ema:
            model_ema.load_state_dict(checkpoint["model_ema"])
        if scaler:
            scaler.load_state_dict(checkpoint["scaler"])

    if args.test_only:
        # We disable the cudnn benchmarking because it can noticeably affect the accuracy
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if model_ema:
            evaluate(model_ema, criterion, data_loader_test, device=device, log_suffix="EMA")
        else:
            evaluate(model, criterion, data_loader_test, device=device)
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args, teacher_model=teacher_model, model_ema=model_ema, scaler=scaler)
        lr_scheduler.step()
        evaluate(model, criterion, data_loader_test, device=device)
        if model_ema:
            evaluate(model_ema, criterion, data_loader_test, device=device, log_suffix="EMA")
        if args.output_dir:
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
            }
            if model_ema:
                checkpoint["model_ema"] = model_ema.state_dict()
            if scaler:
                checkpoint["scaler"] = scaler.state_dict()
            save_on_master(checkpoint, os.path.join(args.output_dir, f"model_{epoch}.pth"))
            save_on_master(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)

    parser.add_argument(
        "--data-path", default=None, type=str, help="dataset path"
    )
    parser.add_argument(
        "--model", default="resnet18", type=str, help="model name"
    )
    parser.add_argument(
        "--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)"
    )
    parser.add_argument(
        "-b", "--batch-size", default=32, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument(
        "--epochs", default=90, type=int, metavar="N", help="number of total epochs to run"
    )
    parser.add_argument(
        "-j", "--workers", default=16, type=int, metavar="N", help="number of data loading workers (default: 16)"
    )
    parser.add_argument(
        "--opt", default="sgd", type=str, help="optimizer"
    )
    parser.add_argument(
        "--lr", default=0.1, type=float, help="initial learning rate"
    )
    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="momentum"
    )
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--norm-weight-decay",
        default=None,
        type=float,
        help="weight decay for Normalization layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--bias-weight-decay",
        default=None,
        type=float,
        help="weight decay for bias parameters of all layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--transformer-embedding-decay",
        default=None,
        type=float,
        help="weight decay for embedding parameters for vision transformer models (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--label-smoothing", default=0.0, type=float, help="label smoothing (default: 0.0)", dest="label_smoothing"
    )
    parser.add_argument(
        "--mixup-alpha", default=0.0, type=float, help="mixup alpha (default: 0.0)"
    )
    parser.add_argument(
        "--cutmix-alpha", default=0.0, type=float, help="cutmix alpha (default: 0.0)"
    )
    parser.add_argument(
        "--lr-scheduler", default="steplr", type=str, help="the lr scheduler (default: steplr)"
    )
    parser.add_argument(
        "--lr-warmup-epochs", default=0, type=int, help="the number of epochs to warmup (default: 0)"
    )
    parser.add_argument(
        "--lr-warmup-method", default="constant", type=str, help="the warmup method (default: constant)"
    )
    parser.add_argument(
        "--lr-warmup-decay", default=0.01, type=float, help="the decay for lr"
    )
    parser.add_argument(
        "--lr-step-size", default=30, type=int, help="decrease lr every step-size epochs"
    )
    parser.add_argument(
        "--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma"
    )
    parser.add_argument(
        "--lr-min", default=0.0, type=float, help="minimum lr of lr schedule (default: 0.0)"
    )
    parser.add_argument(
        "--print-freq", default=50, type=int, help="print frequency"
    )
    parser.add_argument(
        "--output-dir", default=".", type=str, help="path to save outputs"
    )
    parser.add_argument(
        "--resume", default="", type=str, help="path of checkpoint"
    )
    parser.add_argument(
        "--start-epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument(
        "--cache-dataset",
        dest="cache_dataset",
        help="Cache the datasets for quicker initialization. It also serializes the transforms",
        action="store_true",
    )
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--auto-augment", default=None, type=str, help="auto augment policy (default: None)"
    )
    parser.add_argument(
        "--ra-magnitude", default=9, type=int, help="magnitude of auto augment policy"
    )
    parser.add_argument(
        "--augmix-severity", default=3, type=int, help="severity of augmix policy"
    )
    parser.add_argument(
        "--random-erase", default=0.0, type=float, help="random erasing probability (default: 0.0)"
    )

    # Mixed precision training parameters
    parser.add_argument(
        "--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training"
    )

    # distributed training parameters
    parser.add_argument(
        "--world-size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--dist-url", default="env://", type=str, help="url used to set up distributed training"
    )
    parser.add_argument(
        "--model-ema", action="store_true", help="enable tracking Exponential Moving Average of model parameters"
    )
    parser.add_argument(
        "--model-ema-steps",
        type=int,
        default=32,
        help="the number of iterations that controls how often to update the EMA model (default: 32)",
    )
    parser.add_argument(
        "--model-ema-decay",
        type=float,
        default=0.99998,
        help="decay factor for Exponential Moving Average of model parameters (default: 0.99998)",
    )
    parser.add_argument(
        "--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only."
    )
    parser.add_argument(
        "--interpolation", default="bilinear", type=str, help="the interpolation method (default: bilinear)"
    )
    parser.add_argument(
        "--val-resize-size", default=256, type=int, help="the resize size used for validation (default: 256)"
    )
    parser.add_argument(
        "--val-crop-size", default=224, type=int, help="the central crop size used for validation (default: 224)"
    )
    parser.add_argument(
        "--train-crop-size", default=224, type=int, help="the random crop size used for training (default: 224)"
    )
    parser.add_argument(
        "--clip-grad-norm", default=None, type=float, help="the maximum gradient norm (default None)"
    )
    parser.add_argument(
        "--ra-sampler", action="store_true", help="whether to use Repeated Augmentation in training"
    )
    parser.add_argument(
        "--ra-reps", default=3, type=int, help="number of repetitions for Repeated Augmentation (default: 3)"
    )
    parser.add_argument(
        "--weights", default=None, type=str, help="the weights enum name to load"
    )
    parser.add_argument(
        "--backend", default="PIL", type=str.lower, help="PIL or tensor - case insensitive"
    )
    parser.add_argument(
        "--use-v2", action="store_true", help="Use V2 transforms"
    )
    return parser


if __name__ == "__main__":
    """
    Adam, cosine learning rate, L2-norm, grad clipping(1.0)
    Imagenet: batch size=4096/res=224, others: batch size=512/res=128
    mixup: uniform(0, 1)
    inception-style crop
    """
    args = get_args_parser().parse_args()
    main(args)