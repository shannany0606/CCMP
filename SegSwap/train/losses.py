import torch
import torch.nn as nn
import torch.nn.functional as F

def dice_loss_with_logits(input_logits, target_mask):
    input_mask = torch.sigmoid(input_logits)
    mask = input_mask.flatten(start_dim=1)
    gt = target_mask.float().flatten(start_dim=1)
    numerator = 2 * (mask * gt).sum(-1)
    denominator = mask.sum(-1) + gt.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)

    return loss