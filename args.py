from argparse import ArgumentParser


def add_experiment_args(parser: ArgumentParser) -> None:
    # Experiment
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--epochs", default=1200, type=int)
    parser.add_argument("--lr_drop", default=1000, type=int)
    parser.add_argument("--num_classes", default=2, type=int)
    parser.add_argument("--checkpoint_dir", default=None, type=str)
    parser.add_argument("--dataset", default="Prostate", type=str)

    # Model parameters
    parser.add_argument("--model", default="BayeSeg", required=False)
    parser.add_argument("--dataset_dir", default="/workspace/Prostate", type=str)
    parser.add_argument("--in_channels", default=1, type=int)

    # loss weight
    parser.add_argument("--ce_loss_coef", default=1, type=float)
    parser.add_argument("--dice_loss_coef", default=0, type=float)
    parser.add_argument("--bayes_loss_coef", default=100, type=float)


def add_management_args(parser: ArgumentParser) -> None:
    parser.add_argument("--output_dir", default="./logs/model")
    parser.add_argument(
        "--device",
        default="cuda",
        type=str,
        help="device to use for training / testing",
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--start_epoch", default=0, type=int, help="start epoch")
    parser.add_argument("--num_workers", type=int, default=4)


def add_bayes_args(parser: ArgumentParser) -> None:
    # prior hyper-params for appearance mean m
    parser.add_argument("--mu_0", default=0, type=float)
    parser.add_argument("--sigma_0", default=1, type=float)
    # prior hyper-params for appearance std rho
    parser.add_argument("--phi_rho", default=1e-6, type=float)
    parser.add_argument("--gamma_rho", default=2, type=float)
    # Image boundary upsilon
    parser.add_argument("--phi_upsilon", default=1e-8, type=float)
    parser.add_argument("--gamma_upsilon", default=2, type=float)
    # Seg boundary omega
    parser.add_argument("--phi_omega", default=1e-4, type=float)
    parser.add_argument("--gamma_omega", default=2, type=float)
    # Seg category probability pi
    parser.add_argument("--alpha_pi", default=2, type=float)
    parser.add_argument("--beta_pi", default=2, type=float)
