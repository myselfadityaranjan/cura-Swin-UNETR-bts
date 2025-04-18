import os
import torch
import gc
import torch.mps
import matplotlib.pyplot as plt

from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, HausdorffDistanceMetric, ConfusionMatrixMetric
from monai.losses import DiceLoss
from monai.utils.enums import MetricReduction
from monai.data import decollate_batch
from utils.metrics import pad_to_length

import nibabel as nib
import numpy as np

def run_training(
    model,
    loss_fn,
    train_loader,
    val_loader,
    device,
    roi_size,
    save_dir,
):
    scaler = torch.amp.GradScaler()

    dice_metric = DiceMetric( #validation numbers
        include_background=False,
        reduction="mean_batch", #NEED so monai doesnt combine et, wt, tc dice
        get_not_nans=False, #IMPORTANT: or else will show tensor errors
    )

    hd95_metric = HausdorffDistanceMetric(
        include_background=False,
        percentile=95.0,
        reduction="mean_batch",
    )

    sensitivity_metric = ConfusionMatrixMetric(
        include_background=False,
        metric_name="sensitivity",
        reduction="mean_batch",
    )

    specificity_metric = ConfusionMatrixMetric(
        include_background=False,
        metric_name="specificity",
        reduction="mean_batch",
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    train_loss_values = []
    val_dice_et = []
    val_dice_wt = []
    val_dice_tc = []
    val_hd95_et = []
    val_hd95_wt = []
    val_hd95_tc = []
    val_sens_et = []
    val_sens_wt = []
    val_sens_tc = []
    val_spec_et = []
    val_spec_wt = []
    val_spec_tc = []

    max_epochs = 10
    best_metric = 0.0
    best_metric_epoch = -1

    post_sigmoid = torch.nn.Sigmoid()  #TODO: replace w/ Activations(sigmoid=TRUE)
    discrete_mask = torch.nn.Threshold(0.5, 0)  #same thing here

    for epoch in range(max_epochs):
        print(f"\n=== epoch [{epoch+1}/{max_epochs}] ===")
        model.train()
        epoch_loss = 0
        step = 0

        printed_gradients_this_epoch = False

        for batch_data in train_loader:
            step += 1
            images = batch_data['image'].to(device)
            labels = batch_data['label'].to(device)

            optimizer.zero_grad()
            with torch.autocast(device_type='mps', dtype=torch.float16):
                outputs = model(images)
                loss = loss_fn(outputs, labels)

            scaler.scale(loss).backward()

            if not printed_gradients_this_epoch:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        print(f"[epoch {epoch+1}] {name} gradient mean: {param.grad.abs().mean().item():.6f}")
                printed_gradients_this_epoch = True

            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            print(f"  step {step}, train_loss = {loss.item():.4f}")

        epoch_loss /= step
        train_loss_values.append(epoch_loss)
        print(f"epoch {epoch+1} avg loss: {epoch_loss:.4f}")
        scheduler.step()

        model.eval()
        torch.mps.empty_cache()
        gc.collect()

        dice_metric.reset()
        hd95_metric.reset()
        sensitivity_metric.reset()
        specificity_metric.reset()

        print("validation:") #TODO: remove later (debugging to check tensro shape)
        with torch.no_grad():
            for i, val_data in enumerate(val_loader):
                print(f"val batch: {i+1}/{len(val_loader)}")
                print("raw val batch shape", val_data["label"].shape)
                print("raw unique values", torch.unique(val_data["label"]))

                with torch.autocast(device_type="mps", dtype=torch.float16):
                    val_outputs = sliding_window_inference(
                        inputs=val_data["image"].to(device),
                        roi_size=roi_size,
                        sw_batch_size=1,
                        predictor=model,
                        overlap=0.35,
                    )

                val_outputs_list = decollate_batch(val_outputs)
                val_labels_list  = decollate_batch(val_data["label"].to(device))
                val_pred_list    = [discrete_mask(post_sigmoid(x)) for x in val_outputs_list]

                if len(val_labels_list) > 0: #mroe debugging
                    print("Decollated label shape:", val_labels_list[0].shape)
                    for c in range(val_labels_list[0].shape[0]):
                        print(f"Channel {c} in ground truth: Unique values:",
                              torch.unique(val_labels_list[0][c]))

                if len(val_pred_list) > 0:
                    print("Decollated prediction shape:", val_pred_list[0].shape)
                    for c in range(val_pred_list[0].shape[0]):
                        print(f"Channel {c} in prediction: Unique values:",
                              torch.unique(val_pred_list[0][c]))

                val_pred_cpu   = [p.to("cpu") for p in val_pred_list]
                val_labels_cpu = [l.to("cpu") for l in val_labels_list]

                dice_metric(y_pred=val_pred_cpu, y=val_labels_cpu)
                hd95_metric(y_pred=val_pred_cpu, y=val_labels_cpu)
                sensitivity_metric(y_pred=val_pred_cpu, y=val_labels_cpu)
                specificity_metric(y_pred=val_pred_cpu, y=val_labels_cpu)

            dice_vals = dice_metric.aggregate()
            hd95_vals = hd95_metric.aggregate()
            sens_vals = sensitivity_metric.aggregate()
            spec_vals = specificity_metric.aggregate()

            print("raw dice_vals aggregation:", dice_vals)
            print("raw hd95_vals aggregation:", hd95_vals)
            print("raw sens_vals aggregation (type):", type(sens_vals), "value:", sens_vals)
            print("raw spec_vals aggregation (type):", type(spec_vals), "value:", spec_vals)

            if not isinstance(sens_vals, torch.Tensor):
                sens_vals = torch.stack(sens_vals)
            if not isinstance(spec_vals, torch.Tensor):
                spec_vals = torch.stack(spec_vals)

            dice_vals = torch.nan_to_num(dice_vals, nan=0.0)
            hd95_vals = torch.nan_to_num(hd95_vals, nan=0.0)
            sens_vals = torch.nan_to_num(sens_vals, nan=0.0)
            spec_vals = torch.nan_to_num(spec_vals, nan=0.0)

            dice_vals = pad_to_length(dice_vals, 3)
            hd95_vals = pad_to_length(hd95_vals, 3)
            sens_vals = pad_to_length(sens_vals.view(-1), 3)
            spec_vals = pad_to_length(spec_vals.view(-1), 3)

            print("final dice_vals:", dice_vals)
            print("final hd95_vals:", hd95_vals)
            print("final sens_vals:", sens_vals)
            print("final spec_vals:", spec_vals)

            dice_et_val, dice_wt_val, dice_tc_val = [float(v.item()) for v in dice_vals]
            hd95_et_val, hd95_wt_val, hd95_tc_val = [float(v.item()) for v in hd95_vals]
            sens_et_val, sens_wt_val, sens_tc_val = [float(v.item()) for v in sens_vals]
            spec_et_val, spec_wt_val, spec_tc_val = [float(v.item()) for v in spec_vals]

            val_dice_et.append(dice_et_val)
            val_dice_wt.append(dice_wt_val)
            val_dice_tc.append(dice_tc_val)

            val_hd95_et.append(hd95_et_val)
            val_hd95_wt.append(hd95_wt_val)
            val_hd95_tc.append(hd95_tc_val)

            val_sens_et.append(sens_et_val)
            val_sens_wt.append(sens_wt_val)
            val_sens_tc.append(sens_tc_val)

            val_spec_et.append(spec_et_val)
            val_spec_wt.append(spec_wt_val)
            val_spec_tc.append(spec_tc_val)

            mean_dice_val = (dice_et_val + dice_wt_val + dice_tc_val) / 3.0
            print(f"dice (et, wt, tc): {dice_et_val:.4f}, {dice_wt_val:.4f}, {dice_tc_val:.4f}")
            print(f"mean dice: {mean_dice_val:.4f}")

            if mean_dice_val > best_metric:
                best_metric = mean_dice_val
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(save_dir, "swinunetr_model.pth"))
                print(f"  new best: {best_metric:.4f} @ epoch {best_metric_epoch} is saved.")

    print(f"\ntraining finished. best mean dice: {best_metric:.4f} @ epoch {best_metric_epoch}")

    epochs = list(range(1, len(train_loss_values) + 1))

    plt.figure("train_loss", figsize=(8, 6))
    plt.plot(epochs, train_loss_values, label="train_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("epoch avg loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    mean_dice_values = []
    for i in range(len(val_dice_et)):
        mean_val = (val_dice_et[i] + val_dice_wt[i] + val_dice_tc[i]) / 3.0
        mean_dice_values.append(mean_val)

    plt.figure("val avg dice", figsize=(8, 6))
    plt.plot(epochs, mean_dice_values, label="avg dice")
    plt.xlabel("epoch")
    plt.ylabel("dice")
    plt.title("val avg dice (et/wt/tc)")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure("dice-per-channel", figsize=(10, 6))
    plt.plot(epochs, val_dice_et, label="et")
    plt.plot(epochs, val_dice_wt, label="wt")
    plt.plot(epochs, val_dice_tc, label="tc")
    plt.xlabel("epoch")
    plt.ylabel("dice")
    plt.title("valdiation dice by channel")
    plt.legend()
    plt.grid(True)
    plt.show()

    best_epoch_idx = best_metric_epoch - 1
    et_dice_final = val_dice_et[best_epoch_idx]
    wt_dice_final = val_dice_wt[best_epoch_idx]
    tc_dice_final = val_dice_tc[best_epoch_idx]

    et_hd95_final = val_hd95_et[best_epoch_idx]
    wt_hd95_final = val_hd95_wt[best_epoch_idx]
    tc_hd95_final = val_hd95_tc[best_epoch_idx]

    et_sens_final = val_sens_et[best_epoch_idx]
    wt_sens_final = val_sens_wt[best_epoch_idx]
    tc_sens_final = val_sens_tc[best_epoch_idx]

    et_spec_final = val_spec_et[best_epoch_idx]
    wt_spec_final = val_spec_wt[best_epoch_idx]
    tc_spec_final = val_spec_tc[best_epoch_idx]

    print("\n" + "="*50)
    print(f"      final val metrics @ best epoch: {best_metric_epoch}")
    print("="*50)
    print(f"dice:   et={et_dice_final:.4f},wt={wt_dice_final:.4f}, tc={tc_dice_final:.4f}")
    print(f"hausdorff95: et={et_hd95_final:.2f}, wt={wt_hd95_final:.2f}, tc={tc_hd95_final:.2f}")
    print(f"sens:    et={et_sens_final:.4f}, wt={wt_sens_final:.4f}, tc={tc_sens_final:.4f}")
    print(f"spec:    et={et_spec_final:.4f}, wt={wt_spec_final:.4f}, tc={tc_spec_final:.4f}")
    print("="*50)

    return best_metric_epoch, best_metric
