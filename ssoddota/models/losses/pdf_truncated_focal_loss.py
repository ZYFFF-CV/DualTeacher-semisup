"""
Note that:
Some key features are commened during experiments, recover if use it
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import sigmoid_focal_loss as _sigmoid_focal_loss

from mmrotate.models.builder import ROTATED_LOSSES
from mmdet.models.losses.focal_loss import py_sigmoid_focal_loss, py_focal_loss_with_prob, sigmoid_focal_loss
from ssoddota.utils import log_every_n
import logging
import numpy as np
# from .accuracy import accuracy
# from .utils import weight_reduce_loss



@ROTATED_LOSSES.register_module()
class ProbTruncatedFocalLoss(nn.Module):

    def __init__(self,
                 neg_trunc_range = [1,3],
                 pos_trunc_range=1,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0,
                 activated=False):
        """`Focal Loss
        Only used in RPN, note that only 2 calssese are assumed
        Truncate negative/positive instances on both sides

        (1) Truncate positive by 3 sigma rule as the shape is gaussain distributed,
        only the left side that outside the pos_trunc_range*sigma will be excluded

        (2) Truncate negative by two sides, left side the prob of the left bound is
        mean + neg_trunc_ths_prob[0]*sigma, while the right bound is mean + neg_trunc_ths_prob[1]*sigma

        Args:
            neg_trunc_ths (float): threshold where negative scores below will be discarded 
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            alpha (float, optional): A balanced form for Focal Loss.
                Defaults to 0.25.
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'. Options are "none", "mean" and
                "sum".
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
            activated (bool, optional): Whether the input is activated.
                If True, it means the input has been activated and can be
                treated as probabilities. Else, it should be treated as logits.
                Defaults to False.
        """
        super(ProbTruncatedFocalLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.activated = activated
        self.neg_trunc_range = neg_trunc_range
        self.pos_trunc_range = pos_trunc_range

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning label of the prediction.
                The target shape support (N,C) or (N,), (N,C) means
                one-hot form.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """
        target_in = target.clone()

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            if self.activated:
                calculate_loss_func = py_focal_loss_with_prob
            else:
                if pred.dim() == target.dim():
                    # this means that target is already in One-Hot form.
                    calculate_loss_func = py_sigmoid_focal_loss
                elif torch.cuda.is_available() and pred.is_cuda:
                    calculate_loss_func = sigmoid_focal_loss
                else:
                    num_classes = pred.size(1)
                    target = F.one_hot(target, num_classes=num_classes + 1)
                    target = target[:, :num_classes]
                    calculate_loss_func = py_sigmoid_focal_loss
            
            num_classes = pred.size(1)
            # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
            bg_class_ind = num_classes
            pos_inds = ((target_in >= 0)
                    & (target_in < bg_class_ind)).nonzero().reshape(-1)
            neg_inds = (target_in == bg_class_ind).nonzero().reshape(-1)

            pos_inds_valid, neg_inds_valid = pos_inds, neg_inds
            # Just plain

            # pred_sigmoid = pred.sigmoid()

            # truncate FG and BG 

            # Sample by number 
            # sample_pos_num = 256
            # sample_neg_num = 256
            # if len(pos_inds) > sample_pos_num:
            #     pos_inds_valid = self.random_choice(pos_inds, sample_pos_num) #pos_inds[:sample_pos_num]
            # else:
            #     pos_inds_valid = pos_inds
            # neg_inds_valid = self.random_choice(neg_inds, sample_neg_num) #neg_inds[:sample_neg_num]


            # # Hist and prob
            # bin_probs, bin_edges = np.histogram(pred_sigmoid[neg_inds,bg_class_ind-1], 
            #                                  np.arange(100), density=True)
            # # Cummulative prob
            # cumulative_probs = np.cumsum(bin_probs)

            # mean_neg = torch.mean(pred_sigmoid[neg_inds,bg_class_ind-1])
            # sigma_neg = torch.std(pred_sigmoid[neg_inds,bg_class_ind-1])
            # neg_inds_valid = ((pred_sigmoid[neg_inds,bg_class_ind-1] >= mean_neg-self.neg_trunc_range[0] * sigma_neg) & 
            #                   (pred_sigmoid[neg_inds,bg_class_ind-1] <= mean_neg+self.neg_trunc_range[1] * sigma_neg)).nonzero().reshape(-1)
            
            # mean_pos = torch.mean(pred_sigmoid[pos_inds,bg_class_ind-1])
            # sigma_pos = torch.std(pred_sigmoid[pos_inds,bg_class_ind-1])
            # pos_inds_valid = (pred_sigmoid[pos_inds,bg_class_ind-1] >= mean_pos-self.pos_trunc_range * sigma_pos).nonzero().reshape(-1)
            
            # num_neg_valid = len(neg_inds_valid)
            # num_pos_valid = len(pos_inds_valid)
            # log_every_n(
            #     {"num_pos_after": num_pos_valid,"num_neg_after":num_neg_valid,
            #     },
            #     level=logging.INFO
            # )

            # num_neg_valid = 0
            # neg_trunc_ths_curr = self.neg_trunc_ths
            # while num_neg_valid == 0:
            
            #     neg_inds_inds_valid = (pred_sigmoid[neg_inds,bg_class_ind-1] >= neg_trunc_ths_curr).nonzero().reshape(-1)
            #     neg_inds_valid = neg_inds[neg_inds_inds_valid]
            #     num_neg_valid = len(neg_inds_valid)
            #     neg_trunc_ths_curr -= 0.005
                # print(neg_trunc_ths_curr,num_neg_valid)

           
            valid_inds = torch.concat([pos_inds_valid,neg_inds_valid])
            
            
            
            loss_cls = self.loss_weight * calculate_loss_func(
                pred[valid_inds],
                target[valid_inds],
                weight[valid_inds] if weight is not None else weight,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor)

        else:
            raise NotImplementedError
        return loss_cls
    
    def random_choice(self, gallery, num):
        """Random select some elements from the gallery.

        If `gallery` is a Tensor, the returned indices will be a Tensor;
        If `gallery` is a ndarray or list, the returned indices will be a
        ndarray.

        Args:
            gallery (Tensor | ndarray | list): indices pool.
            num (int): expected sample num.

        Returns:
            Tensor or ndarray: sampled indices.
        """
        assert len(gallery) >= num

        is_tensor = isinstance(gallery, torch.Tensor)
        if not is_tensor:
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
            else:
                device = 'cpu'
            gallery = torch.tensor(gallery, dtype=torch.long, device=device)
        # This is a temporary fix. We can revert the following code
        # when PyTorch fixes the abnormal return of torch.randperm.
        # See: https://github.com/open-mmlab/mmdetection/pull/5014
        perm = torch.randperm(gallery.numel())[:num].to(device=gallery.device)
        rand_inds = gallery[perm]
        if not is_tensor:
            rand_inds = rand_inds.cpu().numpy()
        return rand_inds