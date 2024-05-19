import mmcv
import numpy as np
# from mmdet.datasets.pipelines import LoadImageFromFile

from mmrotate.datasets.builder import ROTATED_PIPELINES

@ROTATED_PIPELINES.register_module()
class FilterRotatedAnnotations:
    """Filter invalid annotations.

    Args:
        min_gt_bbox_wh (tuple[int]): Minimum width and height of ground truth
            boxes.
        keep_empty (bool): Whether to return None when it
            becomes an empty bbox after filtering. Default: True
    """

    def __init__(self, min_gt_bbox_wh, keep_empty=True):
        # TODO: add more filter options
        self.min_gt_bbox_wh = min_gt_bbox_wh
        self.keep_empty = keep_empty

    def __call__(self, results):
        assert 'gt_bboxes' in results
        gt_bboxes = results['gt_bboxes']
        if gt_bboxes.shape[0] == 0:
            return results
        w = gt_bboxes[:, 2]
        h = gt_bboxes[:, 3]
        keep = (w > self.min_gt_bbox_wh[0]) & (h > self.min_gt_bbox_wh[1])
        if not keep.any():
            if self.keep_empty:
                return None
            else:
                return results
        else:
            keys = ('gt_bboxes', 'gt_labels', 'gt_masks', 'gt_semantic_seg')
            for key in keys:
                if key in results:
                    results[key] = results[key][keep]
            return results

    def __repr__(self):
        return self.__class__.__name__ + \
               f'(min_gt_bbox_wh={self.min_gt_bbox_wh},' \
               f'always_keep={self.always_keep})'