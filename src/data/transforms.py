import numpy as np
from monai.transforms import (
    Compose,
    LoadImaged,
    ConvertToMultiChannelBasedOnBratsClassesd,
    CropForegroundd,
    RandSpatialCropd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    NormalizeIntensityd,
    EnsureChannelFirstd,
    EnsureTyped,
    RandAffined,
    RandRotated,
    RandGaussianNoised
)

roi_size = (128, 128, 128)

train_transforms = Compose([ #mainly for randomization
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
    CropForegroundd(
        keys=["image", "label"],
        source_key="image",
        k_divisible=roi_size,
        allow_smaller=True
    ),
    RandSpatialCropd(
        keys=["image", "label"],
        roi_size=roi_size,
        random_size=False
    ),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[0, 1, 2]),
    RandAffined(
        keys=["image", "label"],
        rotate_range=(np.pi/12, np.pi/12, np.pi/12),
        scale_range=(0.1, 0.1, 0.1),
        prob=0.5,
        mode=("bilinear", "nearest"),
    ),
    RandRotated(
        keys=["image", "label"],
        range_x=np.pi/12,
        prob=0.3,
        mode=["bilinear", "nearest"],
    ),
    RandGaussianNoised(
        keys=["image"],
        prob=0.3,
        mean=0.0,
        std=0.1,
    ),
    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    RandScaleIntensityd(keys=["image"], factors=0.1, prob=1.0),
    RandShiftIntensityd(keys=["image"], offsets=0.1, prob=1.0),
    EnsureTyped(keys=["image", "label"], dtype=np.float32), #force float32 instead of float64 for mps operations
])

val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    EnsureTyped(keys=["image", "label"], dtype=np.float32), #force float32 again
])
