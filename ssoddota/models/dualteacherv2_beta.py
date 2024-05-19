import torch
from mmcv.runner.fp16_utils import force_fp32
from mmcv.ops import nms_rotated
from mmdet.core import bbox2roi, multi_apply, reduce_mean
from mmdet.models import DETECTORS, build_detector

from ssoddota.utils.structure_utils import dict_split, weighted_loss, dict_sum
from ssoddota.utils import log_image_with_boxes, log_every_n

from .multi_stream_detector import MultiSteamDetector
from .utils import Transform2D, filter_invalid_mmrot#filter_invalid
from mmrotate.core import rbbox2roi, rbbox2result
import numpy as np
from mmrotate.models.builder import build_loss
from mmrotate.models import RotatedSingleStageDetector

from mmcv.cnn import ConvModule
from torch import nn
import logging



@DETECTORS.register_module()
class DualTeacherv2_beta(MultiSteamDetector):
    def __init__(self, model_teacher1: dict, model_teacher2: dict, model_student: dict, train_cfg=None, test_cfg=None, 
                 angle_version=None):
        super(DualTeacherv2_beta, self).__init__(
            dict(teacher_twostage=build_detector(model_teacher1),
                 teacher_onestage=build_detector(model_teacher2), 
                 student=build_detector(model_student)),
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
        if train_cfg is not None:
            # self.freeze("teacher")
            self.freeze("teacher_twostage")
            self.freeze("teacher_onestage")

            self.unsup_weight = self.train_cfg.unsup_weight

            self.unsup_cls_loss_type=train_cfg.unsup_cls_loss_type
            
            if self.unsup_cls_loss_type == 'plain':
                raise TypeError('plain loss is now deleted')
            elif self.unsup_cls_loss_type == 'soft':
                self.unsup_rcnn_cls_func = self.unsup_rcnn_cls_loss_2t1s_soft

            if self.unsup_cls_loss_type != 'plain':
                self.softlearning_after_epoch = train_cfg.softlearning_after_epoch
                self.student_cls_pos_thr = train_cfg.student_cls_pos_thr
        
        assert  angle_version != None
        self.angle_version = angle_version 
        self.epoch = 0
        self.upsup_teacher_pred_weight = 1

        


    def forward_train(self, img, img_metas, **kwargs):
        super().forward_train(img, img_metas, **kwargs)
        kwargs.update({"img": img})
        kwargs.update({"img_metas": img_metas})
        kwargs.update({"tag": [meta["tag"] for meta in img_metas]})
        data_groups = dict_split(kwargs, "tag")
        for _, v in data_groups.items():
            v.pop("tag")

        loss = {}
        #! Warnings: By splitting losses for supervised data and unsupervised data with different names,
        #! it means that at least one sample for each group should be provided on each gpu.
        #! In some situation, we can only put one image per gpu, we have to return the sum of loss
        #! and log the loss with logger instead. Or it will try to sync tensors don't exist.
        if "sup" in data_groups:
            gt_bboxes = data_groups["sup"]["gt_bboxes"]
            log_every_n(
                {"sup_gt_num": sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)}
            )
            sup_loss = self.student.forward_train(**data_groups["sup"])
            sup_loss = {"sup_" + k: v for k, v in sup_loss.items()}
            loss.update(**sup_loss)



            # ################# One stage information for debugging ################# 
            # self.teacher_onestage.eval()
            # feat_onestage = self.teacher_onestage.extract_feat(data_groups["sup"]['img'])
            # teacher_onestage_outs = self.teacher_onestage.bbox_head(feat_onestage)
            # teacher_onestage_pred_list = self.teacher_onestage.bbox_head.get_bboxes(
            #     *teacher_onestage_outs, data_groups["sup"]['img_metas'], rescale=False)

            # teacher_onestage_pred_bboxes, teacher_onestage_pred_labels, _ = multi_apply(
            #     filter_invalid_mmrot,
            #     [bbox[:, :5] for bbox, _ in teacher_onestage_pred_list],
            #     [pred_label for _, pred_label in teacher_onestage_pred_list],
            #     [bbox[:, 5] for bbox, _ in teacher_onestage_pred_list],
            #     thr=self.student_cls_pos_thr,
            #     min_size=self.train_cfg.min_pseduo_box_size,
            # )
            # teacher_onestage_pred_labels = [label.to(teacher_onestage_pred_bboxes[0].device) 
            #                                for label in teacher_onestage_pred_labels] 

            # ################# One stage finished ################# 



        if "unsup_student" in data_groups:
            unsup_loss = weighted_loss(
                self.foward_unsup_train(
                    data_groups["unsup_teacher"], data_groups["unsup_student"]
                ),
                weight=self.unsup_weight,
            )
            unsup_loss = {"unsup_" + k: v for k, v in unsup_loss.items()}
            loss.update(**unsup_loss)

        return loss      


    def foward_unsup_train(self, teacher_data, student_data):
        # sort the teacher and student input to avoid some bugs
        tnames = [meta["filename"] for meta in teacher_data["img_metas"]]
        snames = [meta["filename"] for meta in student_data["img_metas"]]
        tidx = [tnames.index(name) for name in snames]

        with torch.no_grad():
            teacher_info = self.extract_teacher_info(
                teacher_data["img"][
                    torch.Tensor(tidx).to(teacher_data["img"].device).long()
                ],
                [teacher_data["img_metas"][idx] for idx in tidx],
                [teacher_data["proposals"][idx] for idx in tidx]
                if ("proposals" in teacher_data)
                and (teacher_data["proposals"] is not None)
                else None,
            )
        student_info = self.extract_student_info(**student_data)

        return self.compute_pseudo_label_loss(student_info, teacher_info) 
    

    def compute_pseudo_label_loss(self, student_info, teacher_info):
        M = self._get_trans_mat(
            teacher_info["transform_matrix"], student_info["transform_matrix"]
        )
       
        pseudo_bboxes = self._transform_bbox(
            teacher_info["det_bboxes"],
            M,
            [meta["img_shape"] for meta in student_info["img_metas"]],
            self.angle_version
        )
        pseudo_labels = teacher_info["det_labels"]
        loss = {}
        
        
        proposals = student_info["proposals"] # will be 'None' for 1 stage stage detector


        
        loss.update(
            self.unsup_rcnn_cls_func(
                student_info["backbone_feature"],
                ################
                # student_info["backbone_feature_onestage_teacher"],
                ################
                student_info["img_metas"],
                proposals,
                pseudo_bboxes,
                pseudo_labels,
                teacher_info["transform_matrix"],
                student_info["transform_matrix"],
                teacher_info["img_metas"],
                teacher_info["backbone_feature"],
                student_info=student_info,
            )
        )
        return loss
    

    

    def unsup_rcnn_cls_loss_2t1s_soft(
        self,
        feat,
        # feat_extra,
        img_metas,
        proposal_list,
        pseudo_bboxes,
        pseudo_labels,
        teacher_transMat,
        student_transMat,
        teacher_img_metas,
        teacher_feat,
        student_info=None,
        **kwargs,
    ):
        """unsup_rcnn_cls_loss for 2 stage teacher and 1 stage student
        Act as plan supervised laerning, i.e. student use teacher prediction
        as ground truth
        """
        ####### filter low quality bbox for two stage teacher #######
        if self.epoch < self.softlearning_after_epoch:
            gt_bboxes, gt_labels, _ = multi_apply(
                filter_invalid_mmrot,
                [bbox[:, :5] for bbox in pseudo_bboxes],
                pseudo_labels,
                [bbox[:, 5] for bbox in pseudo_bboxes],
                thr=self.train_cfg.cls_pseudo_threshold,
            )
        else: gt_bboxes, gt_labels = pseudo_bboxes, pseudo_labels
        # gt_bboxes, gt_labels, _ = multi_apply(
        #     filter_invalid,
        #     [bbox[:, :4] for bbox in pseudo_bboxes],
        #     pseudo_labels,
        #     [bbox[:, 4] for bbox in pseudo_bboxes],
        #     thr=self.train_cfg.cls_pseudo_threshold,
        # )
        log_every_n(
            {"rcnn_cls_gt_num": sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)}
        )
        

        ####### Plain FOCS training #######
        student_outs = self.student.bbox_head(feat)
        loss_inputs = student_outs + (gt_bboxes, gt_labels, img_metas)
        losses = self.student.bbox_head.loss(*loss_inputs, gt_bboxes_ignore=None)
            
        if len(gt_bboxes[0]) > 0:
            log_image_with_boxes(
                "rcnn_cls",
                student_info["img"][0],
                gt_bboxes[0],
                bbox_tag="pseudo_label",
                labels=gt_labels[0],
                class_names=self.CLASSES,
                interval=500,
                img_norm_cfg=student_info["img_metas"][0]["img_norm_cfg"],
            )
        return losses

    
    @force_fp32(apply_to=["bboxes", "trans_mat"])
    def _transform_bbox(self, bboxes, trans_mat, max_shape, version):
        #bboxes = Transform2D.transform_bboxes(bboxes, trans_mat, max_shape)
        bboxes = Transform2D.transform_rbboxes(bboxes, trans_mat, max_shape, version)
        return bboxes

    @force_fp32(apply_to=["a", "b"])
    def _get_trans_mat(self, a, b):
        return [bt @ at.inverse() for bt, at in zip(b, a)]

    def extract_student_info(self, img, img_metas, proposals=None, **kwargs):
        student_info = {}
        student_info["img"] = img
        feat = self.student.extract_feat(img)

        student_info["backbone_feature"] = feat
        if hasattr(self.student, 'rpn'):
            rpn_out = self.student.rpn_head(feat)
            student_info["rpn_out"] = list(rpn_out)
        student_info["img_metas"] = img_metas
        student_info["proposals"] = proposals
        student_info["transform_matrix"] = [
            torch.from_numpy(meta["transform_matrix"]).float().to(feat[0][0].device)
            for meta in img_metas
        ]
        return student_info

    def extract_teacher_info(self, img, img_metas, proposals=None, **kwargs):

        teacher_info_two = self.extract_teacher_info_st( img, img_metas, proposals, **kwargs)
       
        if self.epoch < self.softlearning_after_epoch: return teacher_info_two
        
        labels_twostage = teacher_info_two["det_labels"]
        bboxes_twostage = teacher_info_two["det_bboxes"]

        bboxes_twostage_filtered, labels_twostage_filtered, _ = multi_apply(
                filter_invalid_mmrot,
                [bbox[:, :5] for bbox in bboxes_twostage],
                labels_twostage,
                [bbox[:, 5] for bbox in bboxes_twostage],
                thr=self.train_cfg.cls_pseudo_threshold,
            )
        
        det_bboxes_list = []
        det_labels_list = []
        teacher_info_one = self.extract_teacher_info_rfcos( img, img_metas, proposals, **kwargs)
        labels_onestage = teacher_info_one["det_labels"]
        bboxes_onestage = teacher_info_one["det_bboxes"]

        log_every_n(
                {"2stage_box_num": sum([len(bbox) for bbox in bboxes_twostage_filtered]) / len(bboxes_twostage_filtered),
                 "1stage_box_num": sum([len(bbox) for bbox in bboxes_onestage]) / len(bboxes_onestage)},
                 level=logging.INFO
            )
        for img_id in range(len(img_metas)):
            # if (len(bboxes_onestage[img_id]) == 0) and (len(bboxes_twostage_filtered[img_id]) > 10):
            #     raise ValueError("Something might be wrong on 'teacher_onestage'") 
                #possbly that teacher_onestage does not inherit weight from student
           

            bboxes_img = torch.cat([bboxes_twostage_filtered[img_id],bboxes_onestage[img_id]],0)
            labels_img = torch.cat([labels_twostage_filtered[img_id],labels_onestage[img_id]],0)
            scores = torch.ones((len(bboxes_img),)).to(bboxes_img.device)
            
            det_bboxes_img = []
            det_labels_img = []

            if len(bboxes_img) == 0:
                det_bboxes_list.append(bboxes_img)
                det_labels_list.append(labels_img)
                continue

            for cls_id in torch.unique(labels_img):
                # slelect each category
                cls_mask = labels_img == cls_id
                cls_boxes = bboxes_img[cls_mask]
                cls_scores = scores[cls_mask]

                # NMS
                keep = nms_rotated(cls_boxes, cls_scores, iou_threshold= self.test_cfg.dual_teacher_nms.iou_thr)
                bbox_keep = keep[0][:,:-1]

                # keep
                det_bboxes_img.append(bbox_keep)
                det_labels_img.append(torch.full((len(bbox_keep),), cls_id, dtype=torch.int64).cuda())

            # concat Tensors
            det_bboxes = torch.cat(det_bboxes_img, dim=0)
            det_labels = torch.cat(det_labels_img, dim=0)

            det_bboxes_list.append(det_bboxes)
            det_labels_list.append(det_labels)

        teacher_info_two["det_labels"] = det_labels_list
        teacher_info_two["det_bboxes"] = det_bboxes_list

        return teacher_info_two
        # if self.epoch < self.softlearning_after_epoch: 
        #     return self.extract_teacher_info_st( img, img_metas, proposals, **kwargs)
        # else:
        #     # print('teacher is rfcos')
        #     return self.extract_teacher_info_rfcos( img, img_metas, proposals, **kwargs)
        
    def extract_teacher_info_rfcos(self, img, img_metas, proposals=None, **kwargs):

        """Extract bbox from rotate fcos
        """
        teacher_info = {}

        ################# One stage information ################# 
        self.teacher_onestage.eval()
        feat_onestage = self.teacher_onestage.extract_feat(img)
        teacher_info["backbone_feature"] = feat_onestage
        teacher_onestage_outs = self.teacher_onestage.bbox_head(feat_onestage)
        teacher_onestage_pred_list = self.teacher_onestage.bbox_head.get_bboxes(
            *teacher_onestage_outs, img_metas, rescale=False)


        # teacher_onestage_pred_bboxes
        ######## Filter will be used in self.extrac_teaehr_info() in the future ############
        # This part may cause error if teacher_onestage is not well trained, as teacher_onestage_pred_bboxes may
        # returns [0,5] tensor
        
        # if self.teacher_onestage.__class__.__name__ == 'RotatedYOLOX':
            
        teacher_onestage_pred_bboxes, teacher_onestage_pred_labels, _ = multi_apply(
            filter_invalid_mmrot,
            [bbox[:, :5] for bbox, _ in teacher_onestage_pred_list],
            [pred_label for _, pred_label in teacher_onestage_pred_list],
            [bbox[:, 5] if bbox.shape[1]>5 else torch.tensor([],dtype=bbox.dtype,device=bbox.device) for bbox, _ in teacher_onestage_pred_list],
            thr=self.student_cls_pos_thr,
            min_size=self.train_cfg.min_pseduo_box_size,
        )
        # else:            
            # teacher_onestage_pred_bboxes, teacher_onestage_pred_labels, _ = multi_apply(
            #     filter_invalid_mmrot,
            #     [bbox[:, :5] for bbox, _ in teacher_onestage_pred_list],
            #     [pred_label for _, pred_label in teacher_onestage_pred_list],
            #     [bbox[:, 5] for bbox, _ in teacher_onestage_pred_list],
            #     thr=self.student_cls_pos_thr,
            #     min_size=self.train_cfg.min_pseduo_box_size,
            # )
        teacher_onestage_pred_labels = [label.to(teacher_onestage_pred_bboxes[0].device) 
                                       for label in teacher_onestage_pred_labels] 

        ################# One stage finished ################# 
        
        teacher_info["proposals"] = proposals
        teacher_info["det_bboxes"] = teacher_onestage_pred_bboxes
        teacher_info["det_labels"] = teacher_onestage_pred_labels
        teacher_info["transform_matrix"] = [
            torch.from_numpy(meta["transform_matrix"]).float().to(feat_onestage[0][0].device)
            for meta in img_metas
        ]

        teacher_info["img_metas"] = img_metas
        return teacher_info
    
    def extract_teacher_info_st(self, img, img_metas, proposals=None, **kwargs):
        teacher_info = {}
        
        feat = self.teacher_twostage.extract_feat(img)
        
        teacher_info["backbone_feature"] = feat
        if proposals is None:
            proposal_cfg = self.teacher_twostage.train_cfg.get(
                "rpn_proposal", self.teacher_twostage.test_cfg.rpn
            )
            rpn_out = list(self.teacher_twostage.rpn_head(feat))
            proposal_list = self.teacher_twostage.rpn_head.get_bboxes(
                *rpn_out, img_metas=img_metas, cfg=proposal_cfg
            )
        else:
            proposal_list = proposals
        teacher_info["proposals"] = proposal_list

        proposal_list, proposal_label_list = self.teacher_twostage.roi_head.simple_test_bboxes(
            feat, img_metas, proposal_list, self.teacher_twostage.test_cfg.rcnn, rescale=False
        )

        proposal_list = [p.to(feat[0].device) for p in proposal_list]
        # proposal_list = [
        #     p if p.shape[0] > 0 else p.new_zeros(0, 5) for p in proposal_list
        # ]
        proposal_list = [
            p if p.shape[0] > 0 else p.new_zeros(0, 6) for p in proposal_list
        ]
        proposal_label_list = [p.to(feat[0].device) for p in proposal_label_list]
        # filter invalid box roughly
        if isinstance(self.train_cfg.pseudo_label_initial_score_thr, float):
            thr = self.train_cfg.pseudo_label_initial_score_thr
        else:
            # TODO: use dynamic threshold
            raise NotImplementedError("Dynamic Threshold is not implemented yet.")
        proposal_list, proposal_label_list, _ = list(
            zip(
                *[
                    filter_invalid_mmrot(
                        proposal,
                        proposal_label,
                        proposal[:, -1],
                        thr=thr,
                        min_size=self.train_cfg.min_pseduo_box_size,
                    )
                    for proposal, proposal_label in zip(
                        proposal_list, proposal_label_list
                    )
                ]
            )
        )
        # proposal_list, proposal_label_list, _ = list(
        #     zip(
        #         *[
        #             filter_invalid(
        #                 proposal,
        #                 proposal_label,
        #                 proposal[:, -1],
        #                 thr=thr,
        #                 min_size=self.train_cfg.min_pseduo_box_size,
        #             )
        #             for proposal, proposal_label in zip(
        #                 proposal_list, proposal_label_list
        #             )
        #         ]
        #     )
        # )
        det_bboxes = proposal_list
        
        reg_unc = self.compute_uncertainty_with_aug(
            feat, img_metas, proposal_list, proposal_label_list
        )
        det_bboxes = [
            torch.cat([bbox, unc], dim=-1) for bbox, unc in zip(det_bboxes, reg_unc)
        ]
        det_labels = proposal_label_list
        teacher_info["det_bboxes"] = det_bboxes
        teacher_info["det_labels"] = det_labels
        teacher_info["transform_matrix"] = [
            torch.from_numpy(meta["transform_matrix"]).float().to(feat[0][0].device)
            for meta in img_metas
        ]
        teacher_info["img_metas"] = img_metas

        return teacher_info

    
    def compute_uncertainty_with_aug(
        self, feat, img_metas, proposal_list, proposal_label_list
    ):
        auged_proposal_list = self.aug_box_mmrot(
            proposal_list, self.train_cfg.jitter_times, self.train_cfg.jitter_scale,
            self.train_cfg.angle_range
        )
        # auged_proposal_list = self.aug_box(
        #     proposal_list, self.train_cfg.jitter_times, self.train_cfg.jitter_scale
        # )
        # flatten
        auged_proposal_list = [
            auged.reshape(-1, auged.shape[-1]) for auged in auged_proposal_list
        ] #(jitter_times,num_bbox,6) --> (jitter_times*num_bbox,6)

        bboxes, _ = self.teacher_twostage.roi_head.simple_test_bboxes(
            feat,
            img_metas,
            auged_proposal_list,
            None,
            rescale=False,
        ) # a list, contains the boxes of the corresponding image in a batch, each 
        # tensor has the shape (num_boxes, 6) and last dimension,
        # 6 represent (cx, cy, w, h, a, score)
        reg_channel = max([bbox.shape[-1] for bbox in bboxes]) // 5
        bboxes = [
            bbox.reshape(self.train_cfg.jitter_times, -1, bbox.shape[-1])
            if bbox.numel() > 0
            else bbox.new_zeros(self.train_cfg.jitter_times, 0, 5 * reg_channel).float()
            for bbox in bboxes
        ]
        # reg_channel = max([bbox.shape[-1] for bbox in bboxes]) // 4
        # bboxes = [
        #     bbox.reshape(self.train_cfg.jitter_times, -1, bbox.shape[-1])
        #     if bbox.numel() > 0
        #     else bbox.new_zeros(self.train_cfg.jitter_times, 0, 4 * reg_channel).float()
        #     for bbox in bboxes
        # ]

        box_unc = [bbox.std(dim=0) for bbox in bboxes]
        bboxes = [bbox.mean(dim=0) for bbox in bboxes]
        # scores = [score.mean(dim=0) for score in scores]
        if reg_channel != 1:
            # bboxes = [
            #     bbox.reshape(bbox.shape[0], reg_channel, 4)[
            #         torch.arange(bbox.shape[0]), label
            #     ]
            #     for bbox, label in zip(bboxes, proposal_label_list)
            # ]
            # box_unc = [
            #     unc.reshape(unc.shape[0], reg_channel, 4)[
            #         torch.arange(unc.shape[0]), label
            #     ]
            #     for unc, label in zip(box_unc, proposal_label_list)
            # ]
            bboxes = [
                bbox.reshape(bbox.shape[0], reg_channel, 5)[
                    torch.arange(bbox.shape[0]), label
                ]
                for bbox, label in zip(bboxes, proposal_label_list)
            ]
            box_unc = [
                unc.reshape(unc.shape[0], reg_channel, 5)[
                    torch.arange(unc.shape[0]), label
                ]
                for unc, label in zip(box_unc, proposal_label_list)
            ]

        # box_shape = [(bbox[:, 2:4] - bbox[:, :2]).clamp(min=1.0) for bbox in bboxes]
        box_shape = [bbox[:, 2:4].clamp(min=1.0) for bbox in bboxes]
        # box_rot = [bbox[:, 4].reshape(-1, 1).clamp(min=1.0) for bbox in bboxes]
        box_angle = [bbox[:, 4].reshape(-1, 1).clamp(min=1.0) for bbox in bboxes]
        # relative unc
        # box_unc = [
        #     unc / wh[:, None, :].expand(-1, 2, 2).reshape(-1, 4)
        #     if wh.numel() > 0
        #     else unc
        #     for unc, wh in zip(box_unc, box_shape)
        # ]
        # box_unc = []
        for unc, wh, angle in zip(box_unc, box_shape, box_angle):
            if wh.numel() > 0:
                wh_expand = wh[:, None, :].expand(-1, 2, 2).reshape(-1, 4)
                unc_size, unc_angle = unc[:,:4] / wh_expand, unc[:,4:5] / angle
                box_unc.append(torch.cat([unc_size, unc_angle],dim=-1))
            else: box_unc.append(unc)


        return box_unc

    
    @staticmethod
    def aug_box(boxes, times=1, frac=0.06):
        def _aug_single(box):
            # random translate
            # TODO: random flip or something
            box_scale = box[:, 2:4] - box[:, :2]
            box_scale = (
                box_scale.clamp(min=1)[:, None, :].expand(-1, 2, 2).reshape(-1, 4)
            )
            aug_scale = box_scale * frac  # [n,4]

            offset = (
                torch.randn(times, box.shape[0], 4, device=box.device)
                * aug_scale[None, ...]
            )
            new_box = box.clone()[None, ...].expand(times, box.shape[0], -1)
            return torch.cat(
                [new_box[:, :, :4].clone() + offset, new_box[:, :, 4:]], dim=-1
            )

        return [_aug_single(box) for box in boxes]

    
    @staticmethod
    def aug_box_mmrot(boxes, times=1, wh_frac=0.06, angle_range_deg=0):
        """shift bbox by both angle and wy, however, shift angle by 0degree by dufault
        we assign different jittering range for xy, wh, and angle, respectively
        Args:
            boxes: a list, where each element is a tensor (N,6) in [x y w h a conf]
            times: the number of box augmentation
            wh_frac: the range of bbox jittering scale in perecentage
            angle_range: the range of jittering the rotation angle of bbox in degree
        Returns:
            a list, where each element is a tensor (jitter_times,N,6) in [x y w h a conf]
        """
        def _aug_single(box):
            angle_range = angle_range_deg/180*np.pi # in radian
            angle_scale = angle_range * torch.ones(box.shape[0],1, device=box.device)


            box_scale = box[:,2:4] #[n,4], each row:[w,h]
            box_scale = (
                box_scale.clamp(min=1)[:, None, :].expand(-1, 2, 2).reshape(-1, 4) * wh_frac
            ) #[N,4], each row:[w,h,w,h,]

            aug_scale = torch.cat([box_scale,angle_scale],dim=-1) # [n,5]

            offset = (
                torch.randn(times, box.shape[0], 5, device=box.device)
                * aug_scale[None, ...]
            ) # calculate jittering value

            new_box = box.clone()[None, ...].expand(times, box.shape[0], -1)

            return torch.cat(
                [new_box[:, :, :5].clone() + offset, new_box[:, :, 5:]], dim=-1
            )
        
        return [_aug_single(box) for box in boxes]



    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        if not any(["student" in key or "teacher" in key for key in state_dict.keys()]):
            keys = list(state_dict.keys())
            state_dict.update({"teacher." + k: state_dict[k] for k in keys})
            state_dict.update({"student." + k: state_dict[k] for k in keys})
            for k in keys:
                state_dict.pop(k)

        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
    
    def set_epoch(self, epoch): 
        self.epoch = epoch 

    def rbbox2result_cuda(self, bboxes, labels, num_classes):
        """Convert detection results to a list of numpy arrays.

        Args:
            bboxes (torch.Tensor): shape (n, 6)
            labels (torch.Tensor): shape (n, )
            num_classes (int): class number, including background class

        Returns:
            list(torch.Tensor): bbox results of each class
        """
        if bboxes.shape[0] == 0:
            return [torch.zeros((0, 6), dtype=torch.float32) for _ in range(num_classes)]
        else:
            # bboxes = bboxes.cpu().numpy()
            # labels = labels.cpu().numpy()
            return [bboxes[labels == i, :] for i in range(num_classes)]
        

    # FCOS loss function, borrowed from 'RotatedFCOSHead.loss'
    @force_fp32(
        apply_to=('cls_scores', 'bbox_preds', 'angle_preds', 'centernesses'))
    def rotate_fcos_loss(self,
             cls_scores,
             bbox_preds,
             angle_preds,
             centernesses,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute loss of the head.
        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * 4.
            angle_preds (list[Tensor]): Box angle for each scale level, \
                each is a 4D-tensor, the channel number is num_points * 1.
            centernesses (list[Tensor]): centerness for each scale level, each
                is a 4D-tensor, the channel number is num_points * 1.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert len(cls_scores) == len(bbox_preds) \
               == len(angle_preds) == len(centernesses)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.student.bbox_head.prior_generator.grid_priors(
            featmap_sizes,
            dtype=bbox_preds[0].dtype,
            device=bbox_preds[0].device)
        labels, bbox_targets, angle_targets = self.student.bbox_head.get_targets(
            all_level_points, gt_bboxes, gt_labels)

        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.student.bbox_head.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_angle_preds = [
            angle_pred.permute(0, 2, 3, 1).reshape(-1, 1)
            for angle_pred in angle_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_angle_preds = torch.cat(flatten_angle_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        flatten_angle_targets = torch.cat(angle_targets)
        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.student.bbox_head.num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = torch.tensor(
            len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_pos = max(reduce_mean(num_pos), 1.0)
        loss_cls = self.student.bbox_head.loss_cls(
            flatten_cls_scores, flatten_labels, avg_factor=num_pos)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_angle_preds = flatten_angle_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]
        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_angle_targets = flatten_angle_targets[pos_inds]
        pos_centerness_targets = self.student.bbox_head.centerness_target(pos_bbox_targets)
        # centerness weighted iou loss
        centerness_denorm = max(
            reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)

        if len(pos_inds) > 0:
            pos_points = flatten_points[pos_inds]
            if self.separate_angle:
                bbox_coder = self.h_bbox_coder
            else:
                bbox_coder = self.bbox_coder
                pos_bbox_preds = torch.cat([pos_bbox_preds, pos_angle_preds],
                                           dim=-1)
                pos_bbox_targets = torch.cat(
                    [pos_bbox_targets, pos_angle_targets], dim=-1)
            pos_decoded_bbox_preds = bbox_coder.decode(pos_points,
                                                       pos_bbox_preds)
            pos_decoded_target_preds = bbox_coder.decode(
                pos_points, pos_bbox_targets)
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                weight=pos_centerness_targets,
                avg_factor=centerness_denorm)
            if self.student.bbox_head.separate_angle:
                loss_angle = self.student.bbox_head.loss_angle(
                    pos_angle_preds, pos_angle_targets, avg_factor=num_pos)
            loss_centerness = self.student.bbox_head.loss_centerness(
                pos_centerness, pos_centerness_targets, avg_factor=num_pos)
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_centerness = pos_centerness.sum()
            if self.student.bbox_head.separate_angle:
                loss_angle = pos_angle_preds.sum()

        if self.student.bbox_head.separate_angle:
            return dict(
                loss_cls=loss_cls,
                loss_bbox=loss_bbox,
                loss_angle=loss_angle,
                loss_centerness=loss_centerness)
        else:
            return dict(
                loss_cls=loss_cls,
                loss_bbox=loss_bbox,
                loss_centerness=loss_centerness)

    