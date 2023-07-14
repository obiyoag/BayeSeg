from .dataset import build_Prostate


def build_dataset(image_set, args):
    if args.dataset == "Prostate":
        return build_Prostate(image_set, args)
    raise ValueError(f"dataset {args.dataset} not supported")
