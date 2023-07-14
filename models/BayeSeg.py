import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientunet import get_efficientunet_b2

from .Basic_module import Criterion, Visualization
from .ResNet import ResNet_appearance, ResNet_shape


class BayeSeg(nn.Module):
    def __init__(self, args):
        super(BayeSeg, self).__init__()

        self.args = args
        self.num_classes = args.num_classes

        self.res_shape = ResNet_shape(num_out_ch=2)
        self.res_appear = ResNet_appearance(num_out_ch=2, num_block=6, bn=True)
        self.unet = get_efficientunet_b2(
            out_channels=2 * args.num_classes, pretrained=False
        )

        self.softmax = nn.Softmax(dim=1)

        Dx = torch.zeros([1, 1, 3, 3], dtype=torch.float)
        Dx[:, :, 1, 1] = 1
        Dx[:, :, 1, 0] = Dx[:, :, 1, 2] = Dx[:, :, 0, 1] = Dx[:, :, 2, 1] = -1 / 4
        self.Dx = nn.Parameter(data=Dx, requires_grad=False)

    @staticmethod
    def sample_normal_jit(mu, log_var):
        sigma = torch.exp(log_var / 2)
        eps = mu.mul(0).normal_()
        z = eps.mul_(sigma).add_(mu)
        return z, eps

    def generate_m(self, samples):
        feature = self.res_appear(samples)
        mu_m, log_var_m = torch.chunk(feature, 2, dim=1)
        log_var_m = torch.clamp(log_var_m, -20, 0)
        m, _ = self.sample_normal_jit(mu_m, log_var_m)
        return m, mu_m, log_var_m

    def generate_x(self, samples):
        feature = self.res_shape(samples)
        mu_x, log_var_x = torch.chunk(feature, 2, dim=1)
        log_var_x = torch.clamp(log_var_x, -20, 0)
        x, _ = self.sample_normal_jit(mu_x, log_var_x)
        return x, mu_x, log_var_x

    def generate_z(self, x):
        feature = self.unet(x.repeat(1, 3, 1, 1))
        mu_z, log_var_z = torch.chunk(feature, 2, dim=1)
        log_var_z = torch.clamp(log_var_z, -20, 0)
        z, _ = self.sample_normal_jit(mu_z, log_var_z)
        if self.training:
            return F.gumbel_softmax(z, dim=1), F.gumbel_softmax(mu_z, dim=1), log_var_z
        else:
            return self.softmax(z), self.softmax(mu_z), log_var_z

    def forward(self, samples: torch.Tensor):
        x, mu_x, log_var_x = self.generate_x(samples)
        m, mu_m, log_var_m = self.generate_m(samples)
        z, mu_z, log_var_z = self.generate_z(x)

        K = self.num_classes
        _, _, W, H = samples.shape

        residual = samples - (x + m)
        mu_rho_hat = (2 * self.args.gamma_rho + 1) / (
            residual * residual + 2 * self.args.phi_rho
        )
        # mu_rho_hat = torch.clamp(mu_rho_hat, 1e4, 1e8)

        normalization = torch.sum(mu_rho_hat).detach()
        n, _ = self.sample_normal_jit(m, torch.log(1 / mu_rho_hat))

        # Image line upsilon
        alpha_upsilon_hat = 2 * self.args.gamma_upsilon + K
        difference_x = F.conv2d(mu_x, self.Dx, padding=1)
        beta_upsilon_hat = (
            torch.sum(
                mu_z * (difference_x * difference_x + 2 * torch.exp(log_var_x)),
                dim=1,
                keepdim=True,
            )
            + 2 * self.args.phi_upsilon
        )  # B x 1 x W x H
        mu_upsilon_hat = alpha_upsilon_hat / beta_upsilon_hat
        # mu_upsilon_hat = torch.clamp(mu_upsilon_hat, 1e6, 1e10)

        # Seg boundary omega
        difference_z = F.conv2d(
            mu_z, self.Dx.expand(K, 1, 3, 3), padding=1, groups=K
        )  # B x K x W x H
        alpha_omega_hat = 2 * self.args.gamma_omega + 1
        pseudo_pi = torch.mean(mu_z, dim=(2, 3), keepdim=True)
        beta_omega_hat = (
            pseudo_pi * (difference_z * difference_z + 2 * torch.exp(log_var_z))
            + 2 * self.args.phi_omega
        )
        mu_omega_hat = alpha_omega_hat / beta_omega_hat
        # mu_omega_hat = torch.clamp(mu_omega_hat, 1e2, 1e6)

        # Seg category probability pi
        _, _, W, H = samples.shape
        alpha_pi_hat = self.args.alpha_pi + W * H / 2
        beta_pi_hat = (
            torch.sum(
                mu_omega_hat * (difference_z * difference_z + 2 * torch.exp(log_var_z)),
                dim=(2, 3),
                keepdim=True,
            )
            / 2
            + self.args.beta_pi
        )
        digamma_pi = torch.special.digamma(
            alpha_pi_hat + beta_pi_hat
        ) - torch.special.digamma(beta_pi_hat)

        # compute loss-related
        kl_y = residual * mu_rho_hat.detach() * residual

        kl_mu_z = torch.sum(
            digamma_pi.detach() * difference_z * mu_omega_hat.detach() * difference_z,
            dim=1,
        )
        kl_sigma_z = torch.sum(
            digamma_pi.detach()
            * (2 * torch.exp(log_var_z) * mu_omega_hat.detach() - log_var_z),
            dim=1,
        )

        kl_mu_x = torch.sum(
            difference_x * difference_x * mu_upsilon_hat.detach() * mu_z.detach(), dim=1
        )
        kl_sigma_x = (
            torch.sum(
                2 * torch.exp(log_var_x) * mu_upsilon_hat.detach() * mu_z.detach(),
                dim=1,
            )
            - log_var_x
        )

        kl_mu_m = self.args.sigma_0 * mu_m * mu_m
        kl_sigma_m = self.args.sigma_0 * torch.exp(log_var_m) - log_var_m

        visualize = {
            "shape": torch.concat([x, mu_x, torch.exp(log_var_x / 2)]),
            "appearance": torch.concat([n, m, 1 / mu_rho_hat.sqrt()]),
            "logit": torch.concat(
                [
                    z[:, 1:2, ...],
                    mu_z[:, 1:2, ...],
                    torch.exp(log_var_z / 2)[:, 1:2, ...],
                ]
            ),
            "shape_boundary": mu_upsilon_hat,
            "seg_boundary": mu_omega_hat[:, 1:2, ...],
        }

        pred = z if self.training else mu_z
        out = {
            "pred_masks": pred,
            "kl_y": kl_y,
            "kl_mu_z": kl_mu_z,
            "kl_sigma_z": kl_sigma_z,
            "kl_mu_x": kl_mu_x,
            "kl_sigma_x": kl_sigma_x,
            "kl_mu_m": kl_mu_m,
            "kl_sigma_m": kl_sigma_m,
            "normalization": normalization,
            "rho": mu_rho_hat,
            "omega": mu_omega_hat * digamma_pi,
            "upsilon": mu_upsilon_hat * mu_z,
            "visualize": visualize,
        }
        return out


class BayeSeg_Criterion(Criterion):
    def __init__(self, args):
        super(BayeSeg_Criterion, self).__init__(args)
        self.bayes_loss_coef = args.bayes_loss_coef

    def loss_Bayes(self, outputs):
        N = outputs["normalization"]
        loss_y = torch.sum(outputs["kl_y"]) / N
        loss_mu_m = torch.sum(outputs["kl_mu_m"]) / N
        loss_sigma_m = torch.sum(outputs["kl_sigma_m"]) / N
        loss_mu_x = torch.sum(outputs["kl_mu_x"]) / N
        loss_sigma_x = torch.sum(outputs["kl_sigma_x"]) / N
        loss_mu_z = torch.sum(outputs["kl_mu_z"]) / N
        loss_sigma_z = torch.sum(outputs["kl_sigma_z"]) / N
        loss_Bayes = (
            loss_y
            + loss_mu_m
            + loss_sigma_m
            + loss_mu_x
            + loss_sigma_x
            + loss_mu_z
            + loss_sigma_z
        )

        return loss_Bayes

    def forward(self, pred, grnd):
        loss_dict = {
            "loss_Dice_CE": self.compute_dice_ce_loss(pred["pred_masks"], grnd),
            "Dice": self.compute_dice(pred["pred_masks"], grnd),
            "loss_Bayes": self.loss_Bayes(pred),
            "rho": torch.mean(pred["rho"]),
            "omega": torch.mean(pred["omega"]),
            "upsilon": torch.mean(pred["upsilon"]),
        }
        losses = (
            loss_dict["loss_Dice_CE"] + self.bayes_loss_coef * loss_dict["loss_Bayes"]
        )
        return losses, loss_dict


class BayeSegVis(Visualization):
    def __init__(self):
        super(BayeSegVis, self).__init__()

    def forward(self, inputs, outputs, labels, others, epoch, writer):
        self.save_image(inputs.as_tensor(), "inputs", epoch, writer)
        self.save_image(outputs.float().as_tensor(), "outputs", epoch, writer)
        self.save_image(labels.float().as_tensor(), "labels", epoch, writer)
        for key, value in others.items():
            self.save_image(value.float().as_tensor(), key, epoch, writer)


def build(args):
    model = BayeSeg(args)
    criterion = BayeSeg_Criterion(args)
    visualizer = BayeSegVis()
    return model, criterion, visualizer
