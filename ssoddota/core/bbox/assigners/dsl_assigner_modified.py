# Copyright (c) OpenMMLab. All rights reserved.
# Copied from mmyolo
# The original one is to support mmdet 3.x
# Here we modify it to support mmdet 2.x and suooport yolox head
# With bug not solved
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmrotate.core.bbox.builder import ROTATED_BBOX_ASSIGNERS as TASK_UTILS
from mmdet.core.bbox import AssignResult


from mmdet.core.bbox.iou_calculators import bbox_overlaps
from mmrotate.core.bbox.iou_calculators import rbbox_overlaps


INF = 100000000
EPS = 1.0e-7


def find_inside_points(boxes: Tensor,
                       points: Tensor,
                       box_dim: int = 4,
                       eps: float = 0.01) -> Tensor:
    """Find inside box points in batches. Boxes dimension must be 3.

    Args:
        boxes (Tensor): Boxes tensor. Must be batch input.
            Has shape of (batch_size, n_boxes, box_dim).
        points (Tensor): Points coordinates. Has shape of (n_points, 2).
        box_dim (int): The dimension of box. 4 means horizontal box and
            5 means rotated box. Defaults to 4.
        eps (float): Make sure the points are inside not on the boundary.
            Only use in rotated boxes. Defaults to 0.01.

    Returns:
        Tensor: A BoolTensor indicating whether a point is inside
        boxes. The index has shape of (n_points, batch_size, n_boxes).
    """
    if box_dim == 4:
        # Horizontal Boxes
        lt_ = points[:, None, None] - boxes[..., :2]
        rb_ = boxes[..., 2:] - points[:, None, None]

        deltas = torch.cat([lt_, rb_], dim=-1)
        is_in_gts = deltas.min(dim=-1).values > 0

    elif box_dim == 5:
        # Rotated Boxes
        points = points[:, None, None]
        ctrs, wh, t = torch.split(boxes, [2, 2, 1], dim=-1)
        cos_value, sin_value = torch.cos(t), torch.sin(t)
        matrix = torch.cat([cos_value, sin_value, -sin_value, cos_value],
                           dim=-1).reshape(*boxes.shape[:-1], 2, 2)

        offset = points - ctrs
        offset = torch.matmul(matrix, offset[..., None])
        offset = offset.squeeze(-1)
        offset_x, offset_y = offset[..., 0], offset[..., 1]
        w, h = wh[..., 0], wh[..., 1]
        is_in_gts = (offset_x <= w / 2 - eps) & (offset_x >= - w / 2 + eps) & \
                    (offset_y <= h / 2 - eps) & (offset_y >= - h / 2 + eps)
    else:
        raise NotImplementedError(f'Unsupport box_dim:{box_dim}')

    return is_in_gts


def get_box_center(boxes: Tensor, box_dim: int = 4) -> Tensor:
    """Return a tensor representing the centers of boxes.

    Args:
        boxes (Tensor): Boxes tensor. Has shape of (b, n, box_dim)
        box_dim (int): The dimension of box. 4 means horizontal box and
            5 means rotated box. Defaults to 4.

    Returns:
        Tensor: Centers have shape of (b, n, 2)
    """
    if box_dim == 4:
        # Horizontal Boxes, (x1, y1, x2, y2)
        return (boxes[..., :2] + boxes[..., 2:]) / 2.0
    elif box_dim == 5:
        # Rotated Boxes, (x, y, w, h, a)
        return boxes[..., :2]
    else:
        raise NotImplementedError(f'Unsupported box_dim:{box_dim}')


@TASK_UTILS.register_module()
class DynamicSoftLabelAssigner_mmdet2(nn.Module):
    """Computes matching between predictions and ground truth with dynamic soft
    label assignment.

    Args:
        num_classes (int): number of class
        soft_center_radius (float): Radius of the soft center prior.
            Defaults to 3.0.
        topk (int): Select top-k predictions to calculate dynamic k
            best matches for each gt. Defaults to 13.
        iou_weight (float): The scale factor of iou cost. Defaults to 3.0.
        iou_calculator (ConfigType): Config of overlaps Calculator.
            Defaults to dict(type='BboxOverlaps2D').
        batch_iou (bool): Use batch input when calculate IoU.
            If set to False use loop instead. Defaults to True.
    """

    def __init__(
        self,
        num_classes,
        soft_center_radius = 3.0,
        topk = 13,
        iou_weight = 3.0,
        # iou_calculator = dict(type='BboxOverlaps2D'),
        rotate_iou = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.soft_center_radius = soft_center_radius
        self.topk = topk
        self.iou_weight = iou_weight
        # self.iou_calculator = TASK_UTILS.build(iou_calculator)
        # self.iou_calculator = build_iou_calculator(iou_calculator)
        self.rotate_iou = rotate_iou

    @torch.no_grad()
    def assign(self, 
               pred_scores, 
               priors,
               pred_bboxes, 
               gt_bboxes,
               gt_labels, 
               gt_bboxes_ignore=None,
                ) -> dict:
       
        #-----Modified---------
        num_gt = gt_bboxes.size(0)
        #----------------------
        decoded_bboxes = pred_bboxes
        num_bboxes, box_dim = decoded_bboxes.size()
        batch_size = 1

        #-----Modified---------
        assigned_gt_inds = decoded_bboxes.new_full((num_bboxes, ),
                                                   0,
                                                   dtype=torch.long)
        #----------------------

        if num_gt == 0 or num_bboxes == 0:
            #-----Modified--------------------------------------------
            max_overlaps = decoded_bboxes.new_zeros((num_bboxes, ))
            if num_gt == 0:
            
               
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
                if gt_labels is None:
                    assigned_labels = None
                else:
                    assigned_labels = decoded_bboxes.new_full((num_bboxes, ),
                                                            -1,
                                                            dtype=torch.long)
                return AssignResult(
                    num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)
            #------------------------------------------------------
           

        prior_center = priors[:, :2]
        is_in_gts = find_inside_points(gt_bboxes, prior_center, box_dim)

        pad_bbox_flag = (gt_bboxes.sum(-1, keepdim=True) > 0).float()
        #----------------------
        is_in_gts = is_in_gts * pad_bbox_flag[..., 0][None]
        is_in_gts = is_in_gts.permute(1, 0, 2)
        valid_mask = is_in_gts.sum(dim=-1) > 0
        

        gt_center = get_box_center(gt_bboxes, box_dim)

        strides = priors[..., 2]
        #-----Modified---------
        distance = (priors.unsqueeze(1)[..., :2] - gt_center[None, :, :]).pow(2).sum(-1).sqrt() / strides[:, None]
        #----------------------

        # prevent overflow
        distance = distance * valid_mask.unsqueeze(-1)
        soft_center_prior = torch.pow(10, distance - self.soft_center_radius)

        #-----Modified---------
        if self.rotate_iou:
            pairwise_ious = rbbox_overlaps(decoded_bboxes, gt_bboxes)
        else:
            pairwise_ious = bbox_overlaps(decoded_bboxes, gt_bboxes)
        #----------------------

        iou_cost = -torch.log(pairwise_ious + EPS) * self.iou_weight

        # select the predicted scores corresponded to the gt_labels
        # in mmdet 3.0 shape=[num_imgs, num_prioir, self.cls_out_channels]
        #-----Modified---------
        # pairwise_pred_scores = pred_scores.permute(0, 2, 1) # shape=[num_imgs, self.cls_out_channels, num_prioir]
        pairwise_pred_scores = pred_scores[None].permute(0, 2, 1) # shape=[num_imgs, self.cls_out_channels, num_prioir]
        #----------------------
        idx = torch.zeros([2, batch_size, num_gt], dtype=torch.long)
        idx[0] = torch.arange(end=batch_size).view(-1, 1).repeat(1, num_gt)
        # idx[1] = gt_labels.long().squeeze(-1)
        #-----Modified---------
        idx[1] = gt_labels.unsqueeze(0).long().squeeze(-1)
        pairwise_pred_scores = pairwise_pred_scores[idx[0],
                                                    idx[1]].permute(0, 2, 1).squeeze(0)
        #----------------------
        # classification cost
        scale_factor = pairwise_ious - pairwise_pred_scores.sigmoid()
        pairwise_cls_cost = F.binary_cross_entropy_with_logits(
            pairwise_pred_scores, pairwise_ious,
            reduction='none') * scale_factor.abs().pow(2.0)

        cost_matrix = pairwise_cls_cost + iou_cost + soft_center_prior

        max_pad_value = torch.ones_like(cost_matrix) * INF
        #-----Modified---------
        # cost_matrix = torch.where(valid_mask[..., None].repeat(1, 1, num_gt),
        #                           cost_matrix, max_pad_value)
        cost_matrix = torch.where(valid_mask.squeeze(0)[..., None].repeat(1, num_gt),
                                  cost_matrix, max_pad_value)
        matched_pred_ious, matched_gt_inds = \
            self.dynamic_k_matching(
                cost_matrix, pairwise_ious, num_gt, valid_mask.squeeze(0))

        # convert to AssignResult format
        assigned_gt_inds[valid_mask] = matched_gt_inds + 1
        assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
        assigned_labels[valid_mask] = gt_labels[matched_gt_inds].long()
        max_overlaps = assigned_gt_inds.new_full((num_bboxes, ),
                                                 -INF,
                                                 dtype=torch.float32)
        max_overlaps[valid_mask] = matched_pred_ious
        return AssignResult(
            num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)
        #----------------------


   
    def dynamic_k_matching(self, cost, pairwise_ious, num_gt, valid_mask):
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)
        # select candidate topk ious for dynamic-k calculation
        candidate_topk = min(self.topk, pairwise_ious.size(0))
        topk_ious, _ = torch.topk(pairwise_ious, candidate_topk, dim=0)
        # calculate dynamic k for each gt
        dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[:, gt_idx], k=dynamic_ks[gt_idx], largest=False)
            matching_matrix[:, gt_idx][pos_idx] = 1

        del topk_ious, dynamic_ks, pos_idx

        prior_match_gt_mask = matching_matrix.sum(1) > 1
        if prior_match_gt_mask.sum() > 0:
            cost_min, cost_argmin = torch.min(
                cost[prior_match_gt_mask, :], dim=1)
            matching_matrix[prior_match_gt_mask, :] *= 0
            matching_matrix[prior_match_gt_mask, cost_argmin] = 1
        # get foreground mask inside box and center prior
        fg_mask_inboxes = matching_matrix.sum(1) > 0
        valid_mask[valid_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[fg_mask_inboxes, :].argmax(1)
        matched_pred_ious = (matching_matrix *
                             pairwise_ious).sum(1)[fg_mask_inboxes]
        return matched_pred_ious, matched_gt_inds