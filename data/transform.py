import torch
from monai.data.meta_obj import get_track_meta
from monai.transforms.utils import generate_spatial_bounding_box
from monai.utils.type_conversion import convert_data_type, convert_to_tensor
from monai.transforms import (
    Compose,
    RandAffined,
    RandGaussianNoised,
    MapTransform,
    ToTensord,
    LoadImaged,
    Orientationd,
    CenterSpatialCropd,
    Resized,
    NormalizeIntensityd,
    Spacingd,
    Rand2DElasticd,
)


class ClipHistogram(MapTransform):
    def __init__(self, keys, percentile, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.percentile = percentile

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            img = d[key]
            img = convert_to_tensor(img, track_meta=get_track_meta())
            values, bin_edges = torch.histogram(img, bins=1000)
            cdf = torch.cumsum(values, dim=0)
            cdf = cdf / cdf[-1]
            clip_value = bin_edges[
                torch.ge(cdf, self.percentile).nonzero().min()
            ].item()
            img = img.clamp(max=clip_value)
            img, *_ = convert_data_type(img, dtype=img.dtype)
            d[key] = img

        return d


class Mask2To1d(MapTransform):
    def __init__(self, keys, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            label = d[key]
            label = convert_to_tensor(label, track_meta=get_track_meta())
            label[label == 2] = 1
            label, *_ = convert_data_type(label, dtype=label.dtype)
            d[key] = label
        return d


class FilterOutBackgroundSliced(MapTransform):
    def __init__(self, keys, source_key, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.source_key = source_key

    def __call__(self, data):
        d = dict(data)
        box_start, box_end = generate_spatial_bounding_box(img=d[self.source_key])
        for key in self.key_iterator(d):
            d[key] = d[key][..., box_start[-1]: box_end[-1]]
        return d


class MaskChange(MapTransform):
    def __init__(self, keys, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            label = d[key]
            for idx in [4, 5, 7, 8, 9, 10, 11, 12, 13]:
                label[label == idx] = 0
            label[label == 1] = 4
            label[label == 6] = 1
            d[key] = label
        return d


class FilterSliced(MapTransform):
    def __init__(self, keys, source_key, samples_per_image, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.source_key = source_key
        self.samples_per_image = samples_per_image

    def __call__(self, data):
        d = dict(data)
        box_start, box_end = generate_spatial_bounding_box(img=d[self.source_key])
        slice_idx = torch.randint(box_start[-1], box_end[-1], (self.samples_per_image,))
        ret = [dict(data) for _ in range(self.samples_per_image)]
        for i in range(self.samples_per_image):
            for key in self.key_iterator(d):
                ret[i][key] = ret[i][key][..., slice_idx[i]]
        return ret


volume_transform = Compose(
    [
        LoadImaged(keys=["image", "label"], image_only=False, ensure_channel_first=True),
        Spacingd(
            keys=["image", "label"],
            pixdim=(0.36458, 0.36458, -1),
            mode=("bilinear", "nearest"),
        ),
        ClipHistogram(keys=["image"], percentile=0.995),
        Orientationd(
            keys=["image", "label"], axcodes="PLS"
        ),  # orientation after spacing
        Mask2To1d(keys=["label"]),
        CenterSpatialCropd(keys=["image", "label"], roi_size=[384, 384, -1]),
    ]
)

slice_transform_train = Compose(
    [
        Resized(
            keys=["image", "label"],
            spatial_size=[192, 192],
            mode=("bilinear", "nearest"),
        ),
        RandAffined(
            keys=["image", "label"],
            mode=("bilinear", "nearest"),
            prob=0.5,
            rotate_range=(3.14 / 6, 3.14 / 6),
            scale_range=(0.2, 0.2),
            translate_range=(10, 10),
        ),
        Rand2DElasticd(
            keys=["image", "label"],
            spacing=(20, 20),
            magnitude_range=(1, 2),
            prob=0.5,
            padding_mode="zeros",
            mode=("bilinear", "nearest"),
        ),
        NormalizeIntensityd(keys=["image"]),
        RandGaussianNoised(keys=["image"], prob=0.5, std=0.5),
        ToTensord(keys=["image", "label"]),
    ]
)

slice_transform_valid = Compose(
    [
        Resized(
            keys=["image", "label"],
            spatial_size=[192, 192],
            mode=("bilinear", "nearest"),
        ),
        NormalizeIntensityd(keys=["image"]),
        ToTensord(keys=["image", "label"]),
    ]
)
