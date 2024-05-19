"""Update compared with 'SoftTeacher/ssod/datasets/pseudo_coco.py'
1. The parent dataset is 'DOTADataset' from mmrot, rather than 'COCO'
2. 'load_annotation' is rewirted ,in the original 'preudo_coco.py', to exclude instance with low confidence.
in this script, I also followed this idea. However the original script introduce a new attribute, called 'load_pesudo_targets',
to load pseudo annotations.
I would like to distinguish 'real' and 'pseudo' annotations by the length of a line:
the origianl dota annotation file only contatains 10 elements: x1,y1,x2,y2,x3,y3,x4,y4,class,difficulty.
I would like to add 'conf' in the end of a line. Thus, the length of a line in the preudo annotation file is 11
"""
import copy
import json
import glob
import numpy as np

from mmdet.datasets import DATASETS
from mmrotate.datasets import  DOTADataset

import os
import os.path as osp


from mmcv.ops import nms_rotated
from mmdet.datasets.custom import CustomDataset

from mmrotate.core import poly2obb_np


@DATASETS.register_module()
class PseudoDOTADataset(DOTADataset):
    # def __init__(
    #     self,
    #     ann_file,
    #     pseudo_ann_file,
    #     pipeline,
    #     confidence_threshold=0.9,
    #     classes=None,
    #     data_root=None,
    #     img_prefix="",
    #     seg_prefix=None,
    #     proposal_file=None,
    #     test_mode=False,
    #     filter_empty_gt=True,
    # ):
    #     self.confidence_threshold = confidence_threshold
    #     self.pseudo_ann_file = pseudo_ann_file

    #     super().__init__(
    #         ann_file,
    #         pipeline,
    #         classes,
    #         data_root,
    #         img_prefix,
    #         seg_prefix,
    #         proposal_file,
    #         test_mode=test_mode,
    #         filter_empty_gt=filter_empty_gt,
    #     )

    def __init__(self,
            
            ann_file,
            pipeline,
            confidence_threshold=0.9,
            version='oc',
            difficulty=100,
            **kwargs):
        
        self.confidence_threshold = confidence_threshold
        super(PseudoDOTADataset, self).__init__(ann_file, pipeline, version, difficulty, **kwargs)
    
    def load_annotations(self, ann_folder):
        """
            Args:
                ann_folder: folder that contains DOTA v1 annotations txt files
        """
        cls_map = {c: i
                   for i, c in enumerate(self.CLASSES)
                   }  # in mmdet v2.0 label is 0-based
        ann_files = glob.glob(ann_folder + '/*.txt')
        data_infos = []
        if not ann_files:  # test phase
            ann_files = glob.glob(ann_folder + '/*.png')
            for ann_file in ann_files:
                data_info = {}
                img_id = osp.split(ann_file)[1][:-4]
                img_name = img_id + '.png'
                data_info['filename'] = img_name
                data_info['ann'] = {}
                data_info['ann']['bboxes'] = []
                data_info['ann']['labels'] = []
                data_infos.append(data_info)
        else:
            for ann_file in ann_files:
                data_info = {}
                img_id = osp.split(ann_file)[1][:-4]
                img_name = img_id + '.png'
                data_info['filename'] = img_name
                data_info['ann'] = {}
                gt_bboxes = []
                gt_labels = []
                gt_polygons = []
                gt_bboxes_ignore = []
                gt_labels_ignore = []
                gt_polygons_ignore = []

                if os.path.getsize(ann_file) == 0:
                    continue

                with open(ann_file) as f:
                    s = f.readlines()
                    for si in s:
                        bbox_info = si.split()
                        ###CHeck if pseudo annotation###
                        if len(bbox_info) == 11: 
                            if float(bbox_info[10]) < self.confidence_threshold: continue
                        poly = np.array(bbox_info[:8], dtype=np.float32)
                        try:
                            x, y, w, h, a = poly2obb_np(poly, self.version)
                        except:  # noqa: E722
                            continue
                        cls_name = bbox_info[8]
                        difficulty = int(bbox_info[9])
                        label = cls_map[cls_name]
                        if difficulty > self.difficulty:
                            pass
                        else:
                            gt_bboxes.append([x, y, w, h, a])
                            gt_labels.append(label)
                            gt_polygons.append(poly)

                if gt_bboxes:
                    data_info['ann']['bboxes'] = np.array(
                        gt_bboxes, dtype=np.float32)
                    data_info['ann']['labels'] = np.array(
                        gt_labels, dtype=np.int64)
                    data_info['ann']['polygons'] = np.array(
                        gt_polygons, dtype=np.float32)
                else:
                    data_info['ann']['bboxes'] = np.zeros((0, 5),
                                                          dtype=np.float32)
                    data_info['ann']['labels'] = np.array([], dtype=np.int64)
                    data_info['ann']['polygons'] = np.zeros((0, 8),
                                                            dtype=np.float32)

                if gt_polygons_ignore:
                    data_info['ann']['bboxes_ignore'] = np.array(
                        gt_bboxes_ignore, dtype=np.float32)
                    data_info['ann']['labels_ignore'] = np.array(
                        gt_labels_ignore, dtype=np.int64)
                    data_info['ann']['polygons_ignore'] = np.array(
                        gt_polygons_ignore, dtype=np.float32)
                else:
                    data_info['ann']['bboxes_ignore'] = np.zeros(
                        (0, 5), dtype=np.float32)
                    data_info['ann']['labels_ignore'] = np.array(
                        [], dtype=np.int64)
                    data_info['ann']['polygons_ignore'] = np.zeros(
                        (0, 8), dtype=np.float32)

                data_infos.append(data_info)

        self.img_ids = [*map(lambda x: x['filename'][:-4], data_infos)]
        return data_infos