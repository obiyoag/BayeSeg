import os
import torch
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from pathlib import Path

from monai.metrics import DiceMetric
from monai.visualize.utils import matshow3d, blend_images
from monai.transforms.compose import Compose
from monai.transforms.io.dictionary import LoadImaged
from monai.transforms import (
    Orientationd,
    CenterSpatialCropd,
    Invertd,
    Resized,
    NormalizeIntensityd,
    Spacingd,
    Transposed,
    ToDeviced,
    AsDiscreted,
)

from models import build_model
from args import add_management_args, add_experiment_args, add_bayes_args
from data.transform import Mask2To1d, FilterOutBackgroundSliced, ClipHistogram


class Tester:
    def __init__(self, args):
        self.args = args
        self.checkpoint_dir = args.checkpoint_dir

        self.visualize = False

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

        self.device = torch.device(args.device)

        self.model, _, _ = build_model(args)
        self.model.to(self.device)
        self.model_type = args.model

        checkpoint_path = os.path.join(self.checkpoint_dir, "checkpoint1200.pth")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(checkpoint["model"])

        n_parameters = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print("number of params:{}".format(n_parameters))

        self.model.eval()

        self.preprocess = Compose(
            [
                LoadImaged(
                    keys=["image", "label", "ori_image"],
                    image_only=False,
                    ensure_channel_first=True,
                ),
                FilterOutBackgroundSliced(
                    keys=["image", "label", "ori_image"], source_key="label"
                ),
                Spacingd(
                    keys="image", pixdim=(0.36458, 0.36458, -1), mode=("bilinear")
                ),
                ClipHistogram(keys="image", percentile=0.995),
                Orientationd(
                    keys=["image", "label", "ori_image"], axcodes="PLS"
                ),  # orientation after spacing
                Mask2To1d(keys="label"),
                CenterSpatialCropd(keys="image", roi_size=[384, 384, -1]),
                Resized(keys="image", spatial_size=[192, 192, -1], mode=("trilinear")),
                Transposed(keys="image", indices=[3, 0, 1, 2]),
                NormalizeIntensityd(keys=["image", "ori_image"], channel_wise=True),
                ToDeviced(keys="image", device=self.device),
            ]
        )

        self.post_pred = Compose(
            [
                ToDeviced(keys="pred", device="cpu"),
                AsDiscreted(keys="pred", argmax=True, dim=1),
                Invertd(keys="pred", transform=self.preprocess, orig_keys="image"),
                Orientationd(keys="pred", axcodes="PLS"),  # orientation after spacing
                AsDiscreted(keys=["pred", "label"], to_onehot=args.num_classes, dim=0),
                Transposed(keys=["pred", "label"], indices=[3, 0, 1, 2]),
            ]
        )

        self.dice_metric = DiceMetric(include_background=False, reduction="mean_batch")

    @torch.no_grad()
    def test_prostate(self):
        site_list = ["RUNMC", "BMC", "BIDMC", "HK", "UCL", "I2CVB"]
        results_list = []
        for site in site_list:
            if self.visualize:
                visual_dir = Path(
                    os.path.join("img", self.checkpoint_dir.split("/")[-1], site)
                )
                visual_dir.mkdir(parents=True, exist_ok=True)

            if site == "RUNMC":
                file_paths = glob(
                    os.path.join(
                        self.args.dataset_dir, "Prostate", site, "test", "*.nii.gz"
                    )
                )
            else:
                file_paths = glob(
                    os.path.join(self.args.dataset_dir, "Prostate", site, "*.nii.gz")
                )

            image_paths, label_paths = [], []
            for path in file_paths:
                if path.split("/")[-1][7:10] in ["seg", "Seg"]:
                    label_paths.append(path)
                else:
                    image_paths.append(path)

            image_paths, label_paths = sorted(image_paths), sorted(label_paths)
            path_dicts = [
                {"image": image_path, "label": label_path, "ori_image": image_path}
                for image_path, label_path in zip(image_paths, label_paths)
            ]

            patient_dices = []
            for i, path_dict in enumerate(path_dicts):
                data_dict = self.preprocess(path_dict)

                outputs = self.model(data_dict["image"])
                data_dict["pred"] = outputs["pred_masks"]

                data_dict = self.post_pred(data_dict)

                self.dice_metric(data_dict["pred"], data_dict["label"])
                patient_dices.append(self.dice_metric.aggregate())
                self.dice_metric.reset()

                if i == 0 and self.visualize:
                    # visualize
                    pred = torch.argmax(
                        data_dict["pred"].permute(1, 2, 3, 0), dim=0, keepdim=True
                    )
                    label = torch.argmax(
                        data_dict["label"].permute(1, 2, 3, 0), dim=0, keepdim=True
                    )

                    ret = blend_images(
                        image=data_dict["ori_image"], label=pred + 3 * label, alpha=0.5
                    )
                    matshow3d(
                        ret,
                        figsize=(50, 50),
                        every_n=1,
                        frame_dim=-1,
                        channel_dim=0,
                        show=True,
                    )
                    plt.savefig(os.path.join(visual_dir, "img_lab_pred.png"))

                    img_num = data_dict["pred"].shape[-1]
                    shape = outputs["visualize"]["shape"][:img_num].permute(1, 2, 3, 0)
                    lines = outputs["visualize"]["shape_boundary"].permute(1, 2, 3, 0)
                    omega = outputs["visualize"]["seg_boundary"].permute(1, 2, 3, 0)

                    matshow3d(
                        shape,
                        figsize=(50, 50),
                        every_n=1,
                        frame_dim=-1,
                        show=True,
                        cmap="gray",
                    )
                    plt.savefig(os.path.join(visual_dir, "shape.png"))
                    matshow3d(
                        lines,
                        figsize=(50, 50),
                        every_n=1,
                        frame_dim=-1,
                        show=True,
                        cmap="gray",
                    )
                    plt.savefig(os.path.join(visual_dir, "shape_boundary.png"))
                    matshow3d(
                        omega,
                        figsize=(50, 50),
                        every_n=1,
                        frame_dim=-1,
                        show=True,
                        cmap="gray",
                    )
                    plt.savefig(os.path.join(visual_dir, "seg_boundary.png"))

            # compute dice
            patient_dices = torch.vstack(patient_dices) * 100
            mean = round(patient_dices.mean().item(), 1)
            std = round(patient_dices.std().item(), 1)
            print(site, mean, std)
            results_list.append(mean)

        print(f"Avg {round(np.array(results_list[1:]).mean(), 1)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("BayeSeg testing", allow_abbrev=False)
    add_experiment_args(parser)
    add_management_args(parser)
    add_bayes_args(parser)
    args = parser.parse_args()

    tester = Tester(args)
    tester.test_prostate()
