import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch
import torch.mps #needed for empty_cache() on mac's MPS

from monai.config import print_config #TODO: fix later

from data.dataset import get_loaders, data_dir, save_dir
from data.transforms import train_transforms, val_transforms, roi_size
from model.architecture import SwinUNETRFallback
from model.loss import DiceCrossEntropyLoss
from training.train_loop import run_training
from inference.inference import run_inference_on_test_case

def main():

    print_config

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"using device: {device}")

    train_loader, val_loader, val_files = get_loaders(train_transforms, val_transforms)

    model = SwinUNETRFallback(
        img_size=roi_size,
        in_channels=4,
        out_channels=3,
        feature_size=24,
        use_checkpoint=False,
    ).to(device)

    loss_fn = DiceCrossEntropyLoss()

    # run training
    best_epoch, best_metric = run_training(
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        roi_size=roi_size,
        save_dir=save_dir,
    )

    test_case = val_files[0] #test_case is [0]
    print("inference case:", test_case["image"])

    run_inference_on_test_case(
        best_model_path=f"{save_dir}/swinunetr_model.pth",
        device=device,
        test_case=test_case,
        val_transforms=val_transforms,
        roi_size=roi_size,
    )

if __name__ == "__main__":
    main()
