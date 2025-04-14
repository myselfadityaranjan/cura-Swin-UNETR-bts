import os
import glob

import torch
from monai.data import DataLoader, Dataset

data_dir = "./data/BraTS2021_Training_Data"
save_dir = "./outputs"

def get_brats_file_paths(root_dir):
    all_dirs = sorted(glob.glob(os.path.join(root_dir, "BraTS2021_*")))
    data_dicts = []
    for pdir in all_dirs:
        flair = glob.glob(os.path.join(pdir, "*_flair.nii.gz"))
        t1ce  = glob.glob(os.path.join(pdir, "*_t1ce.nii.gz"))
        t1    = glob.glob(os.path.join(pdir, "*_t1.nii.gz"))
        t2    = glob.glob(os.path.join(pdir, "*_t2.nii.gz"))
        seg   = glob.glob(os.path.join(pdir, "*_seg.nii.gz"))
        if not (flair and t1ce and t1 and t2 and seg):
            continue
        data_dicts.append({
            "image": [flair[0], t1ce[0], t1[0], t2[0]],
            "label": seg[0]
        })
    return data_dicts

def get_loaders(train_transforms, val_transforms, batch_size=1):
    all_data = get_brats_file_paths(data_dir)
    print(f"total items: {len(all_data)}")

    n_total = len(all_data)
    n_val   = int(n_total * 0.2)
    train_files = all_data[:-n_val]
    val_files   = all_data[-n_val:]
    print(f"training: {len(train_files)}, validation: {len(val_files)}")

    train_ds = Dataset(data=train_files, transform=train_transforms)
    val_ds   = Dataset(data=val_files, transform=val_transforms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    print(f"train loader: {len(train_loader)} batches, val loader: {len(val_loader)} batches")

    return train_loader, val_loader, val_files  #return val files for inference later
