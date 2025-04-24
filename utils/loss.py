import torch
import torch.nn as nn


def sdf_diff_loss(pred, label, weight, scale=1.0, l2_loss=True):
    count = pred.shape[0]
    diff = pred - label
    diff_m = diff / scale  # so it's still in m unit
    if l2_loss:
        loss = (weight * (diff_m**2)).sum() / count  # l2 loss
    else:
        loss = (weight * torch.abs(diff_m)).sum() / count  # l1 loss
    return loss


def color_diff_loss(pred, label, weight, weighted=False, l2_loss=False):
    diff = pred - label
    if not weighted:
        weight = 1.0
    else:
        weight = weight.unsqueeze(1)
    if l2_loss:
        loss = (weight * (diff**2)).mean()
    else:
        loss = (weight * torch.abs(diff)).mean()
    return loss


# used by our approach
def sdf_bce_loss(pred, label, sigma, weight, weighted=False, bce_reduction="mean"):
    """ Calculate the binary cross entropy (BCE) loss for SDF supervision
    Args:
        pred (torch.tenosr): batch of predicted SDF values
        label (torch.tensor): batch of the target SDF values
        sigma (float): scale factor for the sigmoid function
        weight (torch.tenosr): batch of the per-sample weight
        weighted (bool, optional): apply the weight or not
        bce_reduction (string, optional): specifies the reduction to apply to the output
    Returns:
        loss (torch.tensor): BCE loss for the batch
    """
    if weighted:
        loss_bce = nn.BCEWithLogitsLoss(reduction=bce_reduction, weight=weight)
    else:
        loss_bce = nn.BCEWithLogitsLoss(reduction=bce_reduction)
    label_op = torch.sigmoid(label / sigma)  # occupancy prob
    loss = loss_bce(pred / sigma, label_op)
    return loss

def sdf_zhong_loss(pred, label, trunc_dist=None, weight=None, weighted=False):
    if not weighted:
        weight = 1.0
    else:
        weight = weight
    loss = torch.zeros_like(label, dtype=label.dtype, device=label.device)
    middle_point = label / 2.0
    middle_point_abs = torch.abs(middle_point)
    shift_difference_abs = torch.abs(pred - middle_point)
    mask = shift_difference_abs > middle_point_abs
    loss[mask] = (shift_difference_abs - middle_point_abs)[
        mask
    ]  # not masked region simply has a loss of zero, masked region L1 loss
    if trunc_dist is not None:
        surface_mask = torch.abs(label) < trunc_dist
        loss[surface_mask] = torch.abs(pred - label)[surface_mask]
    loss *= weight
    return loss.mean()
