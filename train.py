import os
import sys
import time
import json
import torch
import random
import datetime
import argparse
import numpy as np
from pathlib import Path
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from models import build_model
from data import build_dataset
from utils import get_logger, MetricLogger, SmoothedValue
from args import add_management_args, add_experiment_args, add_bayes_args


class Trainer:
    def __init__(self, args):
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger = get_logger(name="BayeSeg", root=self.output_dir)
        self.logger.info(args)

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        cudnn.benchmark = True

        self.writer = SummaryWriter(log_dir=os.path.join(self.output_dir, "summary"))
        self.device = torch.device(args.device)

        self.model, self.criterion, self.visualizer = build_model(args)
        self.model.to(self.device)
        self.logger.info(self.model)

        n_parameters = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        self.logger.info("number of params:{}".format(n_parameters))

        param_dicts = [
            {"params": [p for n, p in self.model.named_parameters() if p.requires_grad]}
        ]

        self.optimizer = torch.optim.AdamW(
            param_dicts, lr=args.lr, weight_decay=args.weight_decay
        )
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.epochs)

        self.logger.info("Building training dataset...")
        dataset_train = build_dataset(image_set="train", args=args)
        self.logger.info("Number of training images: {}".format(len(dataset_train)))

        self.logger.info("Building validation dataset...")
        dataset_val = build_dataset(image_set="val", args=args)
        self.logger.info("Number of validation images: {}".format(len(dataset_val)))

        self.train_loader = DataLoader(
            dataset_train,
            args.batch_size,
            True,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        self.valid_loader = DataLoader(
            dataset_val,
            args.batch_size,
            False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        if args.resume:
            checkpoint = torch.load(args.resume, map_location="cpu")
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = checkpoint["epoch"] + 1

        self.batch_size = args.batch_size
        self.start_epoch = args.start_epoch
        self.epochs = args.epochs
        self.best_dice = None

    def train(self):
        self.logger.info("Start training")
        start_time = time.time()
        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch

            train_stats = self.train_one_epoch()
            test_stats = self.evaluate()
            self.lr_scheduler.step()
            self.dice_score = test_stats["Dice"]
            self.logger.info("dice score:{}".format(self.dice_score))

            self.save_checkpoints()

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"test_{k}": v for k, v in test_stats.items()},
                "epoch": epoch,
            }

            with open(os.path.join(self.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        self.logger.info("Training time {}".format(total_time_str))

    def train_one_epoch(self):
        self.model.train()
        self.criterion.train()

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))

        total_step = len(self.train_loader)
        train_iterator = iter(self.train_loader)
        start_time = time.time()

        for step in range(total_step):
            start = time.time()
            data_dict = next(train_iterator)
            samples = data_dict["image"].to(self.device)
            targets = data_dict["label"].to(self.device)
            datatime = time.time() - start

            outputs = self.model(samples)
            losses, loss_dict = self.criterion(outputs, targets)

            if not torch.isfinite(losses):
                print("Loss is {}, stopping training".format(losses))
                print(loss_dict)
                sys.exit(1)

            self.optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            metric_logger.update(loss=losses.item(), **loss_dict)
            metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])
            itertime = time.time() - start
            metric_logger.log_every(
                step,
                total_step,
                datatime,
                itertime,
                10,
                "Epoch: [{}]".format(self.epoch),
            )

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        self.logger.info(
            "{} Total time: {} ({:.4f} s / it)".format(
                "Epoch: [{}]".format(self.epoch),
                total_time_str,
                total_time / total_step,
            )
        )
        self.logger.info("Averaged stats:")
        self.logger.info(metric_logger)
        stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

        return stats

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        self.criterion.eval()

        metric_logger = MetricLogger(delimiter="  ")

        total_step = len(self.valid_loader)
        valid_iterator = iter(self.valid_loader)
        sample_list, output_list, target_list = [], [], []
        start_time = time.time()

        for step in range(total_step):
            start = time.time()
            data_dict = next(valid_iterator)
            samples = data_dict["image"].to(self.device)
            targets = data_dict["label"].to(self.device)
            datatime = time.time() - start

            outputs = self.model(samples)
            losses, loss_dict = self.criterion(outputs, targets)

            metric_logger.update(loss=losses.item(), **loss_dict)
            metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])
            itertime = time.time() - start
            metric_logger.log_every(
                step,
                total_step,
                datatime,
                itertime,
                10,
                "Epoch: [{}]".format(self.epoch),
            )

            if step % (max(round(total_step / 16.0), 1)) == 0:
                sample_list.append(samples[0])
                output_list.append(
                    torch.argmax(outputs["pred_masks"][0], dim=0, keepdim=True)
                )
                target_list.append(targets[0])

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        self.logger.info(
            "{} Total time: {} ({:.4f} s / it)".format(
                "Test:", total_time_str, total_time / total_step
            )
        )
        self.logger.info("Averaged stats:")
        self.logger.info(metric_logger)
        stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        self.writer.add_scalar("loss_total", stats["loss"], self.epoch)
        self.writer.add_scalar("Dice", stats["Dice"], self.epoch)
        self.writer.add_scalar("loss_Dice_CE", stats["loss_Dice_CE"], self.epoch)

        self.writer.add_scalar("loss_Bayes", stats["loss_Bayes"], self.epoch)
        self.visualizer(
            torch.stack(sample_list),
            torch.stack(output_list),
            torch.stack(target_list),
            outputs["visualize"],
            self.epoch,
            self.writer,
        )

        return stats

    def save_checkpoints(self):
        checkpoint_paths = [os.path.join(self.output_dir, "checkpoint.pth")]

        if self.best_dice is None or self.dice_score > self.best_dice:
            self.best_dice = self.dice_score
            self.logger.info("Update best model!")
            checkpoint_paths.append(
                os.path.join(self.output_dir, "best_checkpoint.pth")
            )

        if (self.epoch + 1) % 100 == 0 and (self.epoch + 1) >= 1000:
            checkpoint_paths.append(
                os.path.join(self.output_dir, f"checkpoint{self.epoch+1:04}.pth")
            )

        checkpoint_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "epoch": self.epoch + 1,
        }

        for path in checkpoint_paths:
            torch.save(checkpoint_dict, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("BayeSeg training", allow_abbrev=False)
    add_experiment_args(parser)
    add_management_args(parser)
    add_bayes_args(parser)
    args = parser.parse_args()
    trainer = Trainer(args)
    trainer.train()
