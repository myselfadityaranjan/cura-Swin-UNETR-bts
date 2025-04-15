import torch
import torch.nn as nn
from monai.losses import DiceLoss

class DiceCrossEntropyLoss(nn.Module): #dice + bce
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss(to_onehot_y=False, sigmoid=True)
        self.ce   = nn.BCEWithLogitsLoss()

    def forward(self, y_pred, y_true):
        return self.dice(y_pred, y_true) + self.ce(y_pred, y_true.float())
