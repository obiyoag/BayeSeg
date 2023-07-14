import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
from monai.transforms import AsDiscrete
from monai.losses import DiceCELoss, DiceLoss
from monai.metrics import compute_dice, do_metric_reduction


class Balanced_DiceCELoss(DiceCELoss):
    def __init__(self, lambda_ce, lambda_dice):
        super().__init__(lambda_ce=lambda_ce, lambda_dice=lambda_dice)
        self.dice = DiceLoss(
            include_background=True, to_onehot_y=True, reduction="none"
        )

    def ce(self, input: torch.Tensor, target: torch.Tensor, weight: torch.Tensor):
        n_pred_ch, n_target_ch = input.shape[1], target.shape[1]
        if n_pred_ch != n_target_ch and n_target_ch == 1:
            target = torch.squeeze(target, dim=1)
            target = target.long()
        elif not torch.is_floating_point(target):
            target = target.to(dtype=input.dtype)

        return F.cross_entropy(input, target, weight)

    def forward(self, input, target):
        _, counts = torch.unique(target, return_counts=True)
        weight = 1 - counts / target.numel()
        dice_loss = (self.dice(input, target).squeeze() * weight).mean()
        ce_loss = self.ce(input, target, weight)
        total_loss = self.lambda_dice * dice_loss + self.lambda_ce * ce_loss

        return total_loss


class Criterion(nn.Module):
    def __init__(self, args):
        super(Criterion, self).__init__()
        self.compute_dice_ce_loss = Balanced_DiceCELoss(
            args.ce_loss_coef, args.dice_loss_coef
        )
        self.post_label = AsDiscrete(to_onehot=args.num_classes, dim=1)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=args.num_classes, dim=1)

    def compute_dice(self, pred, grnd):
        pred, grnd = self.post_pred(pred), self.post_label(grnd)
        dice, _ = do_metric_reduction(
            compute_dice(pred, grnd, include_background=False)
        )
        return dice

    def forward(self, pred, grnd):
        pred = pred["pred_masks"]
        loss_dict = {
            "loss_Dice_CE": self.compute_dice_ce_loss(pred, grnd),
            "Dice": self.compute_dice(pred, grnd),
        }
        return loss_dict["loss_Dice_CE"], loss_dict


class Visualization(nn.Module):
    def __init__(self):
        super(Visualization, self).__init__()

    @staticmethod
    def save_image(image, tag, epoch, writer):
        image = (image - image.min()) / (image.max() - image.min() + 1e-6)
        grid = make_grid(image, nrow=4, pad_value=1)
        writer.add_image(tag, grid, epoch)
