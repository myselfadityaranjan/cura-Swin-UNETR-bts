# cura-Swin-UNETR-bts

<p align="center">
  <img src="images/assets/cura.png" alt="CURA Logo" width="300"/>
</p>

# About the Project

CURA is a brain tumor (glioblastoma) segmentation pipeline based on Swin-UNETR. This architecture combines Swin Transformers as encoders with UNet-like decoders. Swin-UNETR hybrid models have achieved top-tier performance in the BraTS (Brain Tumor Segmentation) challenges, including BraTS 2021, thanks to their ability to model both local detail and global context in 3D medical images.

Swin-UNETR outperforms standard convolutional models by employing the self-attention mechanism to detect subtle anatomical structures, making it specifically well-suited for procedures like multi-label tumor segmentation (ET/WT/TC).

Though this specific project (CURA) is based on MONAI's reference Swin-UNETR pipeline, it is also custom-tuned for Apple Silicon (MPS backend) and low-memory systems. Hence, it uses a slim fallback Swin-UNETR variant, with reduced feature maps and trimmed skip connections, so that it is more developer-friendly on laptops or limited hardware.

Here is an example of the original Swin-UNETR architecture:

<p align="center">
  <img src="images/swin-UNETR-architecture-overview.png" alt="Swin-UNETR basic pipeline" width="1000"/>
</p>

# Getting Started

The BraTS '21 task dataset for this pipeline can be found at: https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1

The files will be downloaded in .nii.gz (NIFTI) files, and organized into 3 subdirectories (BraTS2021_00495, BraTS2021_00621, BraTS2021_Training_Data). In your main directory, make a "data" directory, such that it is organized:

```bash
cura-Swin-UNETR-bts/
└── data/
    ├── BraTS2021_Training_Data/
    ├── BraTS2021_00495/
    └── BraTS2021_00621/
```
Following, create an "outputs" subdirectory in the main directory. This pipeline will create a .pth file (for the saved model weights) that will be saved in this directory. 

The structure should be:

```bash
cura-Swin-UNETR-bts/
├── data/
│   ├── BraTS2021_Training_Data/
│   ├── BraTS2021_00495/
│   └── BraTS2021_00621/
└── outputs/
    └── swinunetr_model.pth
```

### Features

This pipeline contains the features for:

- Optimized for Apple Silicon (MPS): Full pipeline runs using Metal Performance Shaders (MPS) backend, making it accessible on M-series silicon chips.

- Custom Swin UNETR Variant: lightweight fallback implementation using MONAI’s SwinTransformer encoder with reduced upsampling and memory-aware decoding.

- BraTS 2021 Dataset Loader: Automatically parses BraTS '21 directory structure and stacks FLAIR, T1ce, T1, T2 modalities along with segmentation labels.

- 3D Data Augmentation: Includes random flips, affine rotations, scaling, intensity shifts, Gaussian noise, and cropping tailored for volumetric MRI scans.

- Full MONAI Transforms Pipeline: Data loading, normalization, cropping, augmentation, and typing done via MONAI's flexible Compose and dictionary-based transforms.

- Sliding Window Inference: Supports large-volume inference with MONAI’s sliding_window_inference to handle 3D patch-wise predictions.

- Multi-Metric Validation: Tracks Dice score, Hausdorff95, sensitivity, and specificity across three tumor subregions (ET/WT/TC) at every epoch.

- MPS-aware Training: Uses torch.autocast with AMP-style training to reduce memory usage and enable faster convergence.

- Matplotlib Metric Visualization: Generates plots for training loss, per-channel Dice scores, and validation trends to monitor progression.

- Failsafe Metric Padding & Debugging: tensor padding and detailed console debug outputs to mitigate runtime crashes during aggregation.

