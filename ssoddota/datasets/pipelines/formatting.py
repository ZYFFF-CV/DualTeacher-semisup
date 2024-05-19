import numpy as np
from mmdet.datasets import PIPELINES
from mmdet.datasets.pipelines.formatting import Collect, DefaultFormatBundle, to_tensor
from mmcv.parallel import DataContainer as DC

from ssoddota.core import TrimapMasks

@PIPELINES.register_module()
class DefaultFormatBundle_selfsup(DefaultFormatBundle):
    """
    Similar to 'DefaultFormatBundle' in MMDET
    taking masked img into consideration 
    """
    def __init__(self,
                 img_to_float=True,
                 pad_val=dict(img=0, masks=0, seg=255)):
        super().__init__(img_to_float, pad_val)
    
    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with \
                default bundle.
        """

        if 'img' in results:
            img = results['img']
            img = self.process_img(img,results)
            results['img'] = DC(
                img, padding_value=self.pad_val['img'], stack=True)
        if 'masked_img' in results:
            masked_img = results['masked_img']
            masked_img = self.process_img(masked_img)
            results['masked_img'] = DC(
                masked_img, padding_value=self.pad_val['img'], stack=True)
            
        for key in ['proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels']:
            if key not in results:
                continue
            results[key] = DC(to_tensor(results[key]))
        if 'gt_masks' in results:
            results['gt_masks'] = DC(
                results['gt_masks'],
                padding_value=self.pad_val['masks'],
                cpu_only=True)
        if 'gt_semantic_seg' in results:
            results['gt_semantic_seg'] = DC(
                to_tensor(results['gt_semantic_seg'][None, ...]),
                padding_value=self.pad_val['seg'],
                stack=True)
        return results

    def process_img(self,img,results = None):
        if self.img_to_float is True and img.dtype == np.uint8:
            # Normally, image is of uint8 type without normalization.
            # At this time, it needs to be forced to be converted to
            # flot32, otherwise the model training and inference
            # will be wrong. Only used for YOLOX currently .
            img = img.astype(np.float32)
        # add default meta keys
        if results is not None:
            results = self._add_default_meta_keys(results)
        if len(img.shape) < 3:
            img = np.expand_dims(img, -1)
        # To improve the computational speed by by 3-5 times, apply:
        # If image is not contiguous, use
        # `numpy.transpose()` followed by `numpy.ascontiguousarray()`
        # If image is already contiguous, use
        # `torch.permute()` followed by `torch.contiguous()`
        # Refer to https://github.com/open-mmlab/mmdetection/pull/9533
        # for more details
        if not img.flags.c_contiguous:
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            img = to_tensor(img)
        else:
            img = to_tensor(img).permute(2, 0, 1).contiguous()
        return img



@PIPELINES.register_module()
class ExtraAttrs(object):
    def __init__(self, **attrs):
        self.attrs = attrs

    def __call__(self, results):
        for k, v in self.attrs.items():
            assert k not in results
            results[k] = v
        return results


@PIPELINES.register_module()
class ExtraCollect(Collect):
    def __init__(self, *args, extra_meta_keys=[], **kwargs):
        super().__init__(*args, **kwargs)
        self.meta_keys = self.meta_keys + tuple(extra_meta_keys)

@PIPELINES.register_module()
class PseudoSamples(object):
    def __init__(
        self, with_bbox=False, with_mask=False, with_seg=False, fill_value=255
    ):
        """
        Replacing gt labels in original data with fake labels or adding extra fake labels for unlabeled data.
        This is to remove the effect of labeled data and keep its elements aligned with other sample.
        Args:
            with_bbox:
            with_mask:
            with_seg:
            fill_value:
        """
        self.with_bbox = with_bbox
        self.with_mask = with_mask
        self.with_seg = with_seg
        self.fill_value = fill_value

    def __call__(self, results):
        if self.with_bbox:
            results["gt_bboxes"] = np.zeros((0, 5)) #xywha
            results["gt_labels"] = np.zeros((0,))
            if "bbox_fields" not in results:
                results["bbox_fields"] = []
            if "gt_bboxes" not in results["bbox_fields"]:
                results["bbox_fields"].append("gt_bboxes")
        if self.with_mask:
            num_inst = len(results["gt_bboxes"])
            h, w = results["img"].shape[:2]
            results["gt_masks"] = TrimapMasks(
                [
                    self.fill_value * np.ones((h, w), dtype=np.uint8)
                    for _ in range(num_inst)
                ],
                h,
                w,
            )

            if "mask_fields" not in results:
                results["mask_fields"] = []
            if "gt_masks" not in results["mask_fields"]:
                results["mask_fields"].append("gt_masks")
        if self.with_seg:
            results["gt_semantic_seg"] = self.fill_value * np.ones(
                results["img"].shape[:2], dtype=np.uint8
            )
            if "seg_fields" not in results:
                results["seg_fields"] = []
            if "gt_semantic_seg" not in results["seg_fields"]:
                results["seg_fields"].append("gt_semantic_seg")
        return results