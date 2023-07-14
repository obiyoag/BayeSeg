import os
from glob import glob
from monai.data import CacheDataset, PatchDataset
from data.transform import (
    volume_transform,
    slice_transform_train,
    slice_transform_valid,
    FilterSliced,
)


def build_Prostate(image_set, args):
    assert os.path.exists(
        args.dataset_dir
    ), f"provided data path {args.dataset_dir} does not exist"

    file_paths = glob(os.path.join(args.dataset_dir, "RUNMC", image_set, "*.nii.gz"))

    image_paths, label_paths = [], []
    for path in file_paths:
        if path.split("/")[-1][7:10] in ["seg", "Seg"]:
            label_paths.append(path)
        else:
            image_paths.append(path)

    image_paths, label_paths = sorted(image_paths), sorted(label_paths)
    path_dicts = [
        {"image": image_path, "label": label_path}
        for image_path, label_path in zip(image_paths, label_paths)
    ]

    # split train and val set
    if image_set == "train":
        slice_transform = slice_transform_train
    elif image_set == "val":
        slice_transform = slice_transform_valid

    dataset = CacheDataset(
        data=path_dicts, transform=volume_transform, cache_rate=1.0, num_workers=4
    )
    slice_sampler = FilterSliced(
        ["image", "label"], source_key="label", samples_per_image=12
    )
    slice_dataset = PatchDataset(dataset, slice_sampler, 12, slice_transform)
    return slice_dataset
