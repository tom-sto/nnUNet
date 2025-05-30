import torch
from nnunetv2.training.loss.dice import SoftDiceLoss, MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss, TopKLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1
from torch import nn
from distmap import euclidean_signed_transform

class BoundaryLoss(nn.Module):
    """
    A simple implementation of Boundary Loss for binary segmentation.

    This loss function encourages the model to accurately predict the boundaries
    of objects by penalizing discrepancies between the predicted segmentation
    and the ground truth signed distance map (SDM).

    The core idea is based on the formulation: L_BD = sum(predicted_probability * ground_truth_SDM)
    where SDM is negative inside the object, positive outside, and zero at the boundary.
    Minimizing this sum encourages high probabilities where SDM is negative (inside GT)
    and low probabilities where SDM is positive (outside GT).

    Args:
        sigmoid (bool): If True, applies a sigmoid activation to the inputs.
                        Set to True if your model outputs logits.
                        Set to False if your model outputs probabilities (0-1).
        reduction (str): Specifies the reduction to apply to the output:
                         'mean' | 'sum' | 'none'. Default: 'mean'.
                         'mean' will average the loss over the batch.
        class_idx (int, optional): The index of the foreground class in the target tensor.
                                   Only relevant if targets are one-hot encoded and
                                   have more than one channel. Defaults to 1 (assuming
                                   channel 0 is background, channel 1 is foreground).
    """
    def __init__(self, apply_sigmoid: bool = True, reduction: str = 'mean', class_idx: int = 1):
        super(BoundaryLoss, self).__init__()
        self.apply_sigmoid = apply_sigmoid
        self.reduction = reduction
        self.class_idx = class_idx

        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Reduction must be 'mean', 'sum', or 'none', but got {reduction}")

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Boundary Loss.

        Args:
            inputs (torch.Tensor): Raw outputs from the model.
                                   Shape: (N, C, H, W, D) for 3D or (N, C, H, W) for 2D.
                                   Expect C = 2 for binary background vs foreground
            targets (torch.Tensor): Ground truth segmentation masks.
                                    Shape: (N, C, H, W, D) or (N, C, H, W).
                                    Expected to be one-hot encoded if C > 1, or binary if C=1.

        Returns:
            torch.Tensor: The calculated Boundary Loss.
        """
        if self.apply_sigmoid:
            # Apply sigmoid if inputs are raw values
            inputs = torch.sigmoid(inputs)

        inputs = inputs[:, self.class_idx:self.class_idx+1, ...]    # select foreground only

        # Extract the foreground channel from targets
        # Assuming targets are one-hot encoded (N, C, ...) and foreground is at self.class_idx
        # Or if targets are binary (N, 1, ...)
        if targets.shape[1] > 1:
            # For multi-class or one-hot encoded binary, select the foreground class
            targets = targets[:, self.class_idx:self.class_idx+1, ...]

        sdm_batch = -euclidean_signed_transform(targets, ndim=3)
        # replace any inf's with a large number
        sdm_batch[sdm_batch == float('inf')] = 1000

        # Calculate the core Boundary Loss term
        # Element-wise product of predicted probabilities and SDM
        loss: torch.Tensor = inputs * sdm_batch
        # print("Boundary Loss has gradient:", loss.requires_grad)

        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else: # 'none'
            return loss


class DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, weight_bd=1, ignore_label=None,
                 dice_class=SoftDiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_bd = weight_bd
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.bd = BoundaryLoss()

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0
        bd_loss = self.bd(net_output, target_dice) \
            if self.weight_bd != 0 else 0
        
        print("Dice loss:", dc_loss, "\tCE loss:", ce_loss, "\tBD loss:", bd_loss)

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss + self.weight_bd * bd_loss
        return result


class DC_and_BCE_loss(nn.Module):
    def __init__(self, bce_kwargs, soft_dice_kwargs, weight_ce=1, weight_dice=1, use_ignore_label: bool = False,
                 dice_class=MemoryEfficientSoftDiceLoss):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!

        target mut be one hot encoded
        IMPORTANT: We assume use_ignore_label is located in target[:, -1]!!!

        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super(DC_and_BCE_loss, self).__init__()
        if use_ignore_label:
            bce_kwargs['reduction'] = 'none'

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.use_ignore_label = use_ignore_label

        self.ce = nn.BCEWithLogitsLoss(**bce_kwargs)
        self.dc = dice_class(apply_nonlin=torch.sigmoid, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        if self.use_ignore_label:
            # target is one hot encoded here. invert it so that it is True wherever we can compute the loss
            if target.dtype == torch.bool:
                mask = ~target[:, -1:]
            else:
                mask = (1 - target[:, -1:]).bool()
            # remove ignore channel now that we have the mask
            # why did we use clone in the past? Should have documented that...
            # target_regions = torch.clone(target[:, :-1])
            target_regions = target[:, :-1]
        else:
            target_regions = target
            mask = None

        dc_loss = self.dc(net_output, target_regions, loss_mask=mask)
        target_regions = target_regions.float()
        if mask is not None:
            ce_loss = (self.ce(net_output, target_regions) * mask).sum() / torch.clip(mask.sum(), min=1e-8)
        else:
            ce_loss = self.ce(net_output, target_regions)
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class DC_and_topk_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super().__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = TopKLoss(**ce_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = (target != self.ignore_label).bool()
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result
