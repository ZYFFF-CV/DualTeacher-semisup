"""Modification
1. 'rpn_loss', 'unsup_rcnn_cls_loss', 'unsup_rcnn_reg_loss', 'extract_student_info': use 'filter_invalid_mmrot'
2. 'unsup_rcnn_cls_loss': use 'rbbox2roi' to replace 'bbox2roi'
3. '_transform_bbox' use 'Transform2D.transform_rbboxes'
4. Add 'angle_version' args in the SoftTeacher.__init__
5. 'extract_student_info' use rotate bbox
6. Add 'aug_box_mmrot' for rotate bbox, Add 'angle_range' in self.train_cfg.angle_range
7. Update 'compute_uncertainty_with_aug'


Some Notes:
1. Resuts from 'compute_uncertainty_with_aug' is used in "unsup_rcnn_reg_loss" by get mean value
2. 原始Uncertainey 用法: 
    1) 计算jittering后bbox(xyxy形式)的mean, std
    2) 计算bbox宽高
    3) std/宽高
    4) 求均值
   改良方法：
   1) 计算jittering后bbox(ywha形式)的mean, std
   2) 宽高std/宽高, 角度std/角度
   3) 求均值

TODO: angle_version arg for _transform_bbox !!!!
Add self-supervised learning
#TODO(231017):loss recon in config
"""
import torch
from mmcv.runner.fp16_utils import force_fp32
from mmdet.core import bbox2roi, multi_apply
from mmdet.models import DETECTORS, build_detector

from ssoddota.utils.structure_utils import dict_split, weighted_loss
from ssoddota.utils import log_image_with_boxes, log_every_n

from .multi_stream_detector import MultiSteamDetector
from .utils import Transform2D, filter_invalid_mmrot#filter_invalid
from mmrotate.core import rbbox2roi
import numpy as np
from mmrotate.models.builder import build_loss

from mmcv.cnn import ConvModule
from torch import nn

@DETECTORS.register_module()
class SoftTeacher(MultiSteamDetector):
    def __init__(self, model: dict, train_cfg=None, test_cfg=None, 
                 angle_version=None,training_mode='semi',loss_rescon=None):
        super(SoftTeacher, self).__init__(
            dict(teacher=build_detector(model), student=build_detector(model)),
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
        if train_cfg is not None:
            self.freeze("teacher")
            self.unsup_weight = self.train_cfg.unsup_weight
        
        assert  angle_version != None
        self.angle_version = angle_version 
        self.training_mode = training_mode
        if self.training_mode == 'self':
            self.loss_recon = build_loss(loss_rescon)
            # Add neck similar to BYOL, relu-fc-bn order
            self.projector = nn.Sequential( 
                # Conv
                ConvModule(256,256,3,
                           padding=1,
                           norm_cfg=dict(type='SyncBN'),
                           inplace=False,
                           order=("act", "conv", "norm")),
                # Upsample
                torch.nn.Upsample(scale_factor=2, mode='nearest'),
                 # Conv
                ConvModule(256,256,3,
                           padding=1,
                           norm_cfg=dict(type='SyncBN'),
                           inplace=False,
                           order=("act", "conv", "norm")),
                # Upsample
                torch.nn.Upsample(scale_factor=2, mode='nearest'),  
                # Conv no BN        
                ConvModule(256,3,3,
                           padding=1,
                           norm_cfg=None,
                           act_cfg=None,
                           ),
                )
            # Freeze FPN layers that not used
            self.freeze_layer('student',['neck','fpn_convs'],
                              ['1','2','3'])


    def forward_train(self, img, img_metas, **kwargs):
        if self.training_mode == 'semi':
            return self.forward_train_semi_sup(img, img_metas, **kwargs)
        elif self.training_mode == 'self':
            return self.forward_train_self_sup(img, img_metas, **kwargs)
        else: 
            raise Exception("Undefied train mode: {}".format(self.training_mode))
        

    def forward_train_self_sup(self, img, img_metas, **kwargs):
        """Training loss for self-supervised learning
        """
        super().forward_train(img, img_metas, **kwargs)
        kwargs.update({"img": img})
        kwargs.update({"img_metas": img_metas})
        loss = {}
        # with torch.no_grad():
        #     fpn_feats_target = self.teacher.extract_feat(img) # tuple
        mask = kwargs['gt_masks']
        fpn_feats_pred = self.student.extract_feat(kwargs["masked_img"])
        # fpn_feats_target = self.student.extract_feat(img)
        
        log_image_with_boxes(
            "masked_img",
            kwargs["masked_img"][0],
            torch.tensor([-1,-1,0,0,0]).reshape(1,-1),
            bbox_tag="Pseudo",
            interval=100,
            img_norm_cfg=kwargs["img_metas"][0]["img_norm_cfg"],
            )
        log_image_with_boxes(
            "input_img",
            img[0],
            torch.tensor([-1,-1,0,0,0]).reshape(1,-1),
            bbox_tag="Pseudo",
            interval=100,
            img_norm_cfg=kwargs["img_metas"][0]["img_norm_cfg"],
            )

        # for i in range(len(fpn_feats_pred)):
        #     pred = self.projector(fpn_feats_pred[i])
        #     target = self.projector(fpn_feats_target[i])
        pred = self.projector(fpn_feats_pred[0])
        target = img#self.projector(fpn_feats_target[0])

        loss_reconstruction = self.loss_recon(
            pred,target) #currently only C2
        loss.update({"loss_layer_"+str(0):loss_reconstruction})

        log_image_with_boxes(
            "recovered_img",
            pred[0],
            torch.tensor([-1,-1,0,0,0]).reshape(1,-1),
            bbox_tag="Pseudo",
            interval=100,
            img_norm_cfg=kwargs["img_metas"][0]["img_norm_cfg"],
            )
        return loss

    def freeze_layer(self, model_ref: str, module_ref: list, freeze_ref: list):
        """Freeze specified layers
        args:
            module_ref: a list, to search layer recursively
            freeze_ref: a list, to freeze
        """
        assert model_ref in self.submodules
        submodule = getattr(self, model_ref)
        for module_id in module_ref:
            submodule = getattr(submodule, module_id)
        for freeze_id in freeze_ref:
            layer = getattr(submodule, freeze_id)
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False

        

    def forward_train_semi_sup(self, img, img_metas, **kwargs):
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
            #################################################################
            #在使用30%数据时,捕获runtimeERROR
            try:
                sup_loss = self.student.forward_train(**data_groups["sup"])
            except RuntimeError as e:
                print("Caught a RuntimeError during model inference")
                print("Current batch file names:", [img_meta['filename'] for img_meta in img_metas])
                print("Error details:", e)
            ###################################################################

            sup_loss = {"sup_" + k: v for k, v in sup_loss.items()}
            loss.update(**sup_loss)
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
        rpn_loss, proposal_list = self.rpn_loss(
            student_info["rpn_out"],
            pseudo_bboxes,
            student_info["img_metas"],
            student_info=student_info,
        )
        loss.update(rpn_loss)
        if proposal_list is not None:
            student_info["proposals"] = proposal_list
        

        if self.train_cfg.use_teacher_proposal:
            proposals = self._transform_bbox(
                teacher_info["proposals"],
                M,
                [meta["img_shape"] for meta in student_info["img_metas"]],
                self.angle_version
            )
        else:
            proposals = student_info["proposals"]

        loss.update(
            self.unsup_rcnn_cls_loss(
                student_info["backbone_feature"],
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
        loss.update(
            self.unsup_rcnn_reg_loss(
                student_info["backbone_feature"],
                student_info["img_metas"],
                proposals,
                pseudo_bboxes,
                pseudo_labels,
                student_info=student_info,
            )
        )
        return loss

    
    def rpn_loss(
        self,
        rpn_out,
        pseudo_bboxes,
        img_metas,
        gt_bboxes_ignore=None,
        student_info=None,
        **kwargs,
    ):
        """Use filter_invalid_mmrot
        """
        if self.student.with_rpn:
            gt_bboxes = []
            for bbox in pseudo_bboxes:
                bbox, _, _ = filter_invalid_mmrot(
                    bbox[:, :5],
                    score=bbox[
                        :, 5
                    ],  # TODO: replace with foreground score, here is classification score,
                    thr=self.train_cfg.rpn_pseudo_threshold,
                    min_size=self.train_cfg.min_pseduo_box_size,
                )
                
                # bbox, _, _ = filter_invalid(
                #     bbox[:, :4],
                #     score=bbox[
                #         :, 4
                #     ],  # TODO: replace with foreground score, here is classification score,
                #     thr=self.train_cfg.rpn_pseudo_threshold,
                #     min_size=self.train_cfg.min_pseduo_box_size,
                # )
                gt_bboxes.append(bbox)
            log_every_n(
                {"rpn_gt_num": sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)}
            )
            loss_inputs = rpn_out + [[bbox.float() for bbox in gt_bboxes], img_metas]
            if self.student.rpn_head.__class__.__name__ == 'RotatedFCOSHead_ST':
                loss_inputs = rpn_out + [[bbox.float() for bbox in gt_bboxes], 
                                         [torch.full((len(bbox),), 0, dtype=torch.long, device=bbox.device) for bbox in gt_bboxes], 
                                         img_metas]
            
            losses = self.student.rpn_head.loss(
                *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore
            )
            proposal_cfg = self.student.train_cfg.get(
                "rpn_proposal", self.student.test_cfg.rpn
            )
            proposal_list = self.student.rpn_head.get_bboxes(
                *rpn_out, img_metas=img_metas, cfg=proposal_cfg
            )
            # log_image_with_boxes(
            #     "rpn",
            #     student_info["img"][0],
            #     pseudo_bboxes[0][:, :4],
            #     bbox_tag="rpn_pseudo_label",
            #     scores=pseudo_bboxes[0][:, 4],
            #     interval=500,
            #     img_norm_cfg=student_info["img_metas"][0]["img_norm_cfg"],
            # )
           

            log_image_with_boxes(
                "rpn",
                student_info["img"][0],
                pseudo_bboxes[0][:, :5],
                bbox_tag="rpn_pseudo_label",
                scores=pseudo_bboxes[0][:, 5],
                interval=500,
                img_norm_cfg=student_info["img_metas"][0]["img_norm_cfg"],
            )
            return losses, proposal_list
        else:
            return {}, None

    
    def unsup_rcnn_cls_loss(
        self,
        feat,
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
        gt_bboxes, gt_labels, _ = multi_apply(
            filter_invalid_mmrot,
            [bbox[:, :5] for bbox in pseudo_bboxes],
            pseudo_labels,
            [bbox[:, 5] for bbox in pseudo_bboxes],
            thr=self.train_cfg.cls_pseudo_threshold,
        )
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
        sampling_results = self.get_sampling_result(
            img_metas,
            proposal_list,
            gt_bboxes,
            gt_labels,
        )
        # selected_bboxes = [res.bboxes[:, :4] for res in sampling_results]
        selected_bboxes = [res.bboxes[:, :5] for res in sampling_results]

        #rois = bbox2roi(selected_bboxes)
        rois = rbbox2roi(selected_bboxes) #  Tensor: shape (n, 6), [batch_ind, cx, cy, w, h, a]
        bbox_results = self.student.roi_head._bbox_forward(feat, rois)
        bbox_targets = self.student.roi_head.bbox_head.get_targets(
            sampling_results, gt_bboxes, gt_labels, self.student.train_cfg.rcnn
        )

        M = self._get_trans_mat(student_transMat, teacher_transMat)
        aligned_proposals = self._transform_bbox(
            selected_bboxes,
            M,
            [meta["img_shape"] for meta in teacher_img_metas],
            self.angle_version
        )
        with torch.no_grad():
            _, _scores = self.teacher.roi_head.simple_test_bboxes(
                teacher_feat,
                teacher_img_metas,
                aligned_proposals,
                None,
                rescale=False,
            )
            bg_score = torch.cat([_score[:, -1] for _score in _scores])
            assigned_label, _, _, _ = bbox_targets
            neg_inds = assigned_label == self.student.roi_head.bbox_head.num_classes
            bbox_targets[1][neg_inds] = bg_score[neg_inds].detach()
            
        loss = self.student.roi_head.bbox_head.loss(
            bbox_results["cls_score"],
            bbox_results["bbox_pred"],
            rois,
            *bbox_targets,
            reduction_override="none",
        )
        loss["loss_cls"] = loss["loss_cls"].sum() / max(bbox_targets[1].sum(), 1.0)
        loss["loss_bbox"] = loss["loss_bbox"].sum() / max(
            bbox_targets[1].size()[0], 1.0
        )
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
        return loss

    
    def unsup_rcnn_reg_loss(
        self,
        feat,
        img_metas,
        proposal_list,
        pseudo_bboxes,
        pseudo_labels,
        student_info=None,
        **kwargs,
    ):
        gt_bboxes, gt_labels, _ = multi_apply(
            filter_invalid_mmrot,
            [bbox[:, :5] for bbox in pseudo_bboxes],
            pseudo_labels,
            [-bbox[:, 6:].mean(dim=-1) for bbox in pseudo_bboxes],
            thr=-self.train_cfg.reg_pseudo_threshold,
        )
        # gt_bboxes, gt_labels, _ = multi_apply(
        #     filter_invalid,
        #     [bbox[:, :4] for bbox in pseudo_bboxes],
        #     pseudo_labels,
        #     [-bbox[:, 5:].mean(dim=-1) for bbox in pseudo_bboxes],
        #     thr=-self.train_cfg.reg_pseudo_threshold,
        # )
        log_every_n(
            {"rcnn_reg_gt_num": sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)}
        )
        loss_bbox = self.student.roi_head.forward_train(
            feat, img_metas, proposal_list, gt_bboxes, gt_labels, **kwargs
        )["loss_bbox"]
        if len(gt_bboxes[0]) > 0:
            log_image_with_boxes(
                "rcnn_reg",
                student_info["img"][0],
                gt_bboxes[0],
                bbox_tag="pseudo_label",
                labels=gt_labels[0],
                class_names=self.CLASSES,
                interval=500,
                img_norm_cfg=student_info["img_metas"][0]["img_norm_cfg"],
            )
        return {"loss_bbox": loss_bbox}
    
    def get_sampling_result(
        self,
        img_metas,
        proposal_list,
        gt_bboxes,
        gt_labels,
        gt_bboxes_ignore=None,
        **kwargs,
    ):
        num_imgs = len(img_metas)
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(num_imgs)]
        sampling_results = []
        for i in range(num_imgs):
            assign_result = self.student.roi_head.bbox_assigner.assign(
                proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i], gt_labels[i]
            )
            sampling_result = self.student.roi_head.bbox_sampler.sample(
                assign_result,
                proposal_list[i],
                gt_bboxes[i],
                gt_labels[i],
            )
            sampling_results.append(sampling_result)
        return sampling_results

    
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
        if self.student.with_rpn:
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
        teacher_info = {}
        feat = self.teacher.extract_feat(img)
        teacher_info["backbone_feature"] = feat
        if proposals is None:
            proposal_cfg = self.teacher.train_cfg.get(
                "rpn_proposal", self.teacher.test_cfg.rpn
            )
            rpn_out = list(self.teacher.rpn_head(feat))
            proposal_list = self.teacher.rpn_head.get_bboxes(
                *rpn_out, img_metas=img_metas, cfg=proposal_cfg
            )
        else:
            proposal_list = proposals
        teacher_info["proposals"] = proposal_list

        proposal_list, proposal_label_list = self.teacher.roi_head.simple_test_bboxes(
            feat, img_metas, proposal_list, self.teacher.test_cfg.rcnn, rescale=False
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

        bboxes, _ = self.teacher.roi_head.simple_test_bboxes(
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
    