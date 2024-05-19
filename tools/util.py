import sys

import cv2
import matplotlib.pyplot as plt
import mmcv
import numpy as np
import pycocotools.mask as mask_util
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

from mmdet.core.evaluation.panoptic_utils import INSTANCE_OFFSET
from mmdet.core.mask.structures import bitmap_to_polygon
from mmdet.core.utils import mask2ndarray
from mmdet.core.visualization.palette import get_palette, palette_val
from mmrotate.core.visualization.image import imshow_det_rbboxes

import glob
import os
import os.path as osp
from mmrotate.core import poly2obb_np

def imshow_gt_det_rbboxes(img,
                         annotation,
                         result,
                         class_names=None,
                         score_thr=0,
                         gt_bbox_color=(61, 102, 255),
                         gt_text_color=(200, 200, 200),
                         gt_mask_color=(61, 102, 255),
                         det_bbox_color=(241, 101, 72),
                         det_text_color=(200, 200, 200),
                         det_mask_color=(241, 101, 72),
                         thickness=2,
                         font_size=13,
                         win_name='',
                         show=True,
                         wait_time=0,
                         out_file=None,
                         overlay_gt_pred=True):
    """General visualization GT and result function, optimized for rbbox

    Args:
      img (str | ndarray): The image to be displayed.
      annotation (dict): Ground truth annotations where contain keys of
          'gt_bboxes' and 'gt_labels' or 'gt_masks'.
      result (tuple[list] | list): The detection result, can be either
          (bbox, segm) or just bbox.
      class_names (list[str]): Names of each classes.
      score_thr (float): Minimum score of bboxes to be shown. Default: 0.
      gt_bbox_color (list[tuple] | tuple | str | None): Colors of bbox lines.
          If a single color is given, it will be applied to all classes.
          The tuple of color should be in RGB order. Default: (61, 102, 255).
      gt_text_color (list[tuple] | tuple | str | None): Colors of texts.
          If a single color is given, it will be applied to all classes.
          The tuple of color should be in RGB order. Default: (200, 200, 200).
      gt_mask_color (list[tuple] | tuple | str | None, optional): Colors of
          masks. If a single color is given, it will be applied to all classes.
          The tuple of color should be in RGB order. Default: (61, 102, 255).
      det_bbox_color (list[tuple] | tuple | str | None):Colors of bbox lines.
          If a single color is given, it will be applied to all classes.
          The tuple of color should be in RGB order. Default: (241, 101, 72).
      det_text_color (list[tuple] | tuple | str | None):Colors of texts.
          If a single color is given, it will be applied to all classes.
          The tuple of color should be in RGB order. Default: (200, 200, 200).
      det_mask_color (list[tuple] | tuple | str | None, optional): Color of
          masks. If a single color is given, it will be applied to all classes.
          The tuple of color should be in RGB order. Default: (241, 101, 72).
      thickness (int): Thickness of lines. Default: 2.
      font_size (int): Font size of texts. Default: 13.
      win_name (str): The window name. Default: ''.
      show (bool): Whether to show the image. Default: True.
      wait_time (float): Value of waitKey param. Default: 0.
      out_file (str, optional): The filename to write the image.
          Default: None.
      overlay_gt_pred (bool): Whether to plot gts and predictions on the
       same image. If False, predictions and gts will be plotted on two same
       image which will be concatenated in vertical direction. The image
       above is drawn with gt, and the image below is drawn with the
       prediction result. Default: True.

    Returns:
        ndarray: The image with bboxes or masks drawn on it.
    """
    assert 'gt_bboxes' in annotation
    assert 'gt_labels' in annotation
    assert isinstance(result, (tuple, list, dict)), 'Expected ' \
        f'tuple or list or dict, but get {type(result)}'

    gt_bboxes = annotation['gt_bboxes']
    gt_labels = annotation['gt_labels']
    gt_masks = annotation.get('gt_masks', None)
    if gt_masks is not None:
        gt_masks = mask2ndarray(gt_masks)

    gt_seg = annotation.get('gt_semantic_seg', None)
    if gt_seg is not None:
        pad_value = 255  # the padding value of gt_seg
        sem_labels = np.unique(gt_seg)
        all_labels = np.concatenate((gt_labels, sem_labels), axis=0)
        all_labels, counts = np.unique(all_labels, return_counts=True)
        stuff_labels = all_labels[np.logical_and(counts < 2,
                                                 all_labels != pad_value)]
        stuff_masks = gt_seg[None] == stuff_labels[:, None, None]
        gt_labels = np.concatenate((gt_labels, stuff_labels), axis=0)
        gt_masks = np.concatenate((gt_masks, stuff_masks.astype(np.uint8)),
                                  axis=0)
        # If you need to show the bounding boxes,
        # please comment the following line
        # gt_bboxes = None

    img = mmcv.imread(img)

    img_with_gt = imshow_det_rbboxes(
        img,
        gt_bboxes,
        gt_labels,
        gt_masks,
        class_names=class_names,
        bbox_color=gt_bbox_color,
        text_color=gt_text_color,
        mask_color=gt_mask_color,
        thickness=thickness,
        font_size=font_size,
        win_name=win_name,
        show=False)

    if not isinstance(result, dict):
        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = result, None

        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)

        segms = None
        if segm_result is not None and len(labels) > 0:  # non empty
            segms = mmcv.concat_list(segm_result)
            segms = mask_util.decode(segms)
            segms = segms.transpose(2, 0, 1)
    else:
        assert class_names is not None, 'We need to know the number ' \
                                        'of classes.'
        VOID = len(class_names)
        bboxes = None
        pan_results = result['pan_results']
        # keep objects ahead
        ids = np.unique(pan_results)[::-1]
        legal_indices = ids != VOID
        ids = ids[legal_indices]
        labels = np.array([id % INSTANCE_OFFSET for id in ids], dtype=np.int64)
        segms = (pan_results[None] == ids[:, None, None])

    if overlay_gt_pred:
        img = imshow_det_rbboxes(
            img_with_gt,
            bboxes,
            labels,
            segms=segms,
            class_names=class_names,
            score_thr=score_thr,
            bbox_color=det_bbox_color,
            text_color=det_text_color,
            mask_color=det_mask_color,
            thickness=thickness,
            font_size=font_size,
            win_name=win_name,
            show=show,
            wait_time=wait_time,
            out_file=out_file)
    else:
        img_with_det = imshow_det_rbboxes(
            img,
            bboxes,
            labels,
            segms=segms,
            class_names=class_names,
            score_thr=score_thr,
            bbox_color=det_bbox_color,
            text_color=det_text_color,
            mask_color=det_mask_color,
            thickness=thickness,
            font_size=font_size,
            win_name=win_name,
            show=False)
        img = np.concatenate([img_with_gt, img_with_det], axis=0)

        plt.imshow(img)
        if show:
            if wait_time == 0:
                plt.show()
            else:
                plt.show(block=False)
                plt.pause(wait_time)
        if out_file is not None:
            mmcv.imwrite(img, out_file)
        plt.close()

    return img

def load_annotations_dota(ann_folder,CLASSES,filter_empty_gt=True,
                          difficulty_th=100,
                          angle_version='le90'):
    """
        Args:
            ann_folder: folder that contains DOTA v1 annotations txt files
            CLASSES: a tuple defining the names of classes
            filter_empty_gt: bool, whether exclude non-foreground annotations,
                    default: True
            difficulty_th: threshold to exclude diffcult examples, default: 100
            angle_version: string, which way to convert xyxy to xywha, default 'le90'
            
    """
    cls_map = {c: i
                for i, c in enumerate(CLASSES)
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

            if os.path.getsize(ann_file) == 0 and filter_empty_gt:
                continue

            with open(ann_file) as f:
                s = f.readlines()
                for si in s:
                    bbox_info = si.split()
                    poly = np.array(bbox_info[:8], dtype=np.float32)
                    try:
                        x, y, w, h, a = poly2obb_np(poly, angle_version)
                    except:  # noqa: E722
                        continue
                    cls_name = bbox_info[8]
                    difficulty = int(bbox_info[9])
                    label = cls_map[cls_name]
                    if difficulty > difficulty_th:
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

    # self.img_ids = [*map(lambda x: x['filename'][:-4], data_infos)]
    return data_infos