import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # Apply sigmoid - IMPORTANT: Your model outputs logits, not probabilities
        pred = torch.sigmoid(pred)

        # Flatten tensors
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        # Calculate Dice coefficient (correctly)
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice  # This should be positive


class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=0.5, ce_weight=0.5, smooth=1.0):
        super(CombinedLoss, self).__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice_loss = DiceLoss(smooth=smooth)
        self.ce_loss = nn.BCEWithLogitsLoss()  # This applies sigmoid internally

    def forward(self, pred, target):
        # Ensure target is float for BCE
        target = target.float()

        # Calculate losses
        dice_loss = self.dice_loss(pred, target)
        ce_loss = self.ce_loss(pred, target)

        # Combine losses (both should be positive)
        combined_loss = self.dice_weight * dice_loss + self.ce_weight * ce_loss

        # Debugging
        # print(
        #     f"Dice loss: {dice_loss.item()}, BCE loss: {ce_loss.item()}, Combined: {combined_loss.item()}")

        return combined_loss


# Focal Loss for handling class imbalance
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        # Ensure target is float
        target = target.float()

        # Apply sigmoid function to input
        pred = torch.sigmoid(pred)

        # Calculate BCE loss
        bce_loss = F.binary_cross_entropy(pred, target, reduction='none')

        # Calculate focal weight
        pt = target * pred + (1 - target) * (1 - pred)
        focal_weight = (1 - pt) ** self.gamma

        # Apply alpha
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)

        # Calculate focal loss
        focal_loss = alpha_weight * focal_weight * bce_loss

        return focal_loss.mean()
