import torch
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch, DataLoader, Dataset
from monai.transforms import Activations, AsDiscrete

def run_inference_on_test_case(
    best_model_path,
    device,
    test_case,
    val_transforms,
    roi_size,
):
    model_state = torch.load(best_model_path)

    from model.architecture import SwinUNETRFallback
    model = SwinUNETRFallback( #define model shape (fallback; not full SWIN)
        img_size=roi_size,
        in_channels=4,
        out_channels=3,
        feature_size=24,
        use_checkpoint=False,
    ).to(device)
    model.load_state_dict(model_state)
    model.eval()

    test_ds = Dataset(data=[test_case], transform=val_transforms)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    post_sigmoid = Activations(sigmoid=True)
    discrete_mask = AsDiscrete(threshold=0.5)

    with torch.no_grad():
        with torch.autocast(device_type="mps", dtype=torch.float16):
            for batch_data in test_loader:
                outputs = sliding_window_inference(
                    batch_data["image"].to(device), roi_size=roi_size, sw_batch_size=1, predictor=model, overlap=0.35
                )
                out_list = decollate_batch(outputs)
                preds = [discrete_mask(post_sigmoid(x)) for x in out_list]
                tumourmask_3ch = preds[0].cpu().numpy()  #shape: (3, d, h, w)

                slice_idx = tumourmask_3ch.shape[1] // 2
                flair_img_nib = nib.load(test_case["image"][0])
                flair_data = flair_img_nib.get_fdata()

                seg_slice = tumourmask_3ch[:, slice_idx, :, :]
                pred_wt = seg_slice[1]

                plt.figure("test inference", (12, 6))
                plt.subplot(1, 2, 1)
                plt.title("flair slice")
                plt.imshow(flair_data[:, :, slice_idx], cmap="gray")
                plt.subplot(1, 2, 2)
                plt.title("predicted wt (channel=1)")
                plt.imshow(pred_wt, cmap="jet", alpha=0.5)
                plt.show()
