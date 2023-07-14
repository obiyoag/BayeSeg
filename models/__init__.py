from .BayeSeg import build as build_BayeSeg


def build_model(args):
    if args.model == "BayeSeg":
        return build_BayeSeg(args)
    else:
        raise ValueError("invalid model:{}".format(args.model))
