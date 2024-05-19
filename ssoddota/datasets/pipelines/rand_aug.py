"""
Borrowed from https://github.com/microsoft/SoftTeacher/blob/main/ssod/datasets/pipelines/rand_aug.py
Update:
mmdet load COCO in xyxy format, while mmrot load dota in xywha format
1. Use 'transforms' from 'mmrot', instead of mmdet
2. RandTranslate._translate_bboxes is converted from xyxy to xywha, but still Shift bboxes horizontally or vertically, without angle shift
3. RandRotate._rotate_bboxes is converted from xyxy to xywha
4. RandShear._shear_bboxes
5. GeometricAugmentation._filter_invalid
6. Add 'angle_version' in 'GeometricAugmentation' args
7. use RRandomFlip rather than RandomFlip



TODO:
1. Currentluy the bbox can only be shifted by x or y, expected to add roatation in the future
2. The 'randomerase' only add horizontal region, expected to add rotate region in the future
3. 'RecomputeBox' only support mask to xyxy, add rotate bbox in the future
"""
import copy
import random
import torch

import cv2
import mmcv
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from mmcv.image.colorspace import bgr2rgb, rgb2bgr
from mmdet.core.mask import BitmapMasks, PolygonMasks
from mmdet.datasets import PIPELINES
from mmdet.datasets.pipelines import Compose as BaseCompose
from mmdet.datasets.pipelines import transforms
from mmrotate.datasets.pipelines import transforms as rtrans
from mmrotate.core import obb2poly_np, poly2obb_np, poly2obb
from mmrotate.core.bbox.transforms import get_best_begin_point



from .geo_utils import GeometricTransformationBase as GTrans

PARAMETER_MAX = 10


def int_parameter(level, maxval, max_level=None):
    if max_level is None:
        max_level = PARAMETER_MAX
    return int(level * maxval / max_level)


def float_parameter(level, maxval, max_level=None):
    if max_level is None:
        max_level = PARAMETER_MAX
    return float(level) * maxval / max_level


class RandAug(object):
    """refer to https://github.com/google-research/ssl_detection/blob/00d52272f
    61b56eade8d5ace18213cba6c74f6d8/detection/utils/augmentation.py#L240."""

    def __init__(
        self,
        prob: float = 1.0,
        magnitude: int = 10,
        random_magnitude: bool = True,
        record: bool = False,
        magnitude_limit: int = 10,
        angle_version: str = 'oc',
    ):
        assert 0 <= prob <= 1, f"probability should be in (0,1) but get {prob}"
        assert (
            magnitude <= PARAMETER_MAX
        ), f"magnitude should be small than max value {PARAMETER_MAX} but get {magnitude}"

        self.prob = prob
        self.magnitude = magnitude
        self.magnitude_limit = magnitude_limit
        self.random_magnitude = random_magnitude
        self.record = record
        self.buffer = None
        self.angle_version = angle_version

    def __call__(self, results):
        if np.random.random() < self.prob:
            magnitude = self.magnitude
            if self.random_magnitude:
                magnitude = np.random.randint(1, magnitude)
            if self.record:
                if "aug_info" not in results:
                    results["aug_info"] = []
                results["aug_info"].append(self.get_aug_info(magnitude=magnitude))
            results = self.apply(results, magnitude)
        # clear buffer
        return results

    def apply(self, results, magnitude: int = None):
        raise NotImplementedError()

    def __repr__(self):
        return f"{self.__class__.__name__}(prob={self.prob},magnitude={self.magnitude},max_magnitude={self.magnitude_limit},random_magnitude={self.random_magnitude})"

    def get_aug_info(self, **kwargs):
        aug_info = dict(type=self.__class__.__name__)
        aug_info.update(
            dict(
                prob=1.0,
                random_magnitude=False,
                record=False,
                magnitude=self.magnitude,
            )
        )
        aug_info.update(kwargs)
        return aug_info

    def enable_record(self, mode: bool = True):
        self.record = mode

    def set_angle_version(self, version):
        self.angle_version = version

@PIPELINES.register_module()
class Identity(RandAug):
    def apply(self, results, magnitude: int = None):
        return results


@PIPELINES.register_module()
class AutoContrast(RandAug):
    def apply(self, results, magnitude=None):
        for key in results.get("img_fields", ["img"]):
            img = bgr2rgb(results[key])
            results[key] = rgb2bgr(
                np.asarray(ImageOps.autocontrast(Image.fromarray(img)), dtype=img.dtype)
            )
        return results


@PIPELINES.register_module()
class RandEqualize(RandAug):
    def apply(self, results, magnitude=None):
        for key in results.get("img_fields", ["img"]):
            img = bgr2rgb(results[key])
            results[key] = rgb2bgr(
                np.asarray(ImageOps.equalize(Image.fromarray(img)), dtype=img.dtype)
            )
        return results


@PIPELINES.register_module()
class RandSolarize(RandAug):
    def apply(self, results, magnitude=None):
        for key in results.get("img_fields", ["img"]):
            img = results[key]
            results[key] = mmcv.solarize(
                img, min(int_parameter(magnitude, 256, self.magnitude_limit), 255)
            )
        return results

def _enhancer_impl(enhancer):
    """Sets level to be between 0.1 and 1.8 for ImageEnhance transforms of
    PIL."""

    def impl(pil_img, level, max_level=None):
        v = float_parameter(level, 1.8, max_level) + 0.1  # going to 0 just destroys it
        return enhancer(pil_img).enhance(v)

    return impl


class RandEnhance(RandAug):
    op = None

    def apply(self, results, magnitude=None):
        for key in results.get("img_fields", ["img"]):
            img = bgr2rgb(results[key])

            results[key] = rgb2bgr(
                np.asarray(
                    _enhancer_impl(self.op)(
                        Image.fromarray(img), magnitude, self.magnitude_limit
                    ),
                    dtype=img.dtype,
                )
            )
        return results


@PIPELINES.register_module()
class RandColor(RandEnhance):
    op = ImageEnhance.Color


@PIPELINES.register_module()
class RandContrast(RandEnhance):
    op = ImageEnhance.Contrast


@PIPELINES.register_module()
class RandBrightness(RandEnhance):
    op = ImageEnhance.Brightness


@PIPELINES.register_module()
class RandSharpness(RandEnhance):
    op = ImageEnhance.Sharpness


@PIPELINES.register_module()
class RandPosterize(RandAug):
    def apply(self, results, magnitude=None):
        for key in results.get("img_fields", ["img"]):
            img = bgr2rgb(results[key])
            magnitude = int_parameter(magnitude, 4, self.magnitude_limit)
            results[key] = rgb2bgr(
                np.asarray(
                    ImageOps.posterize(Image.fromarray(img), 4 - magnitude),
                    dtype=img.dtype,
                )
            )
        return results


@PIPELINES.register_module()
class Sequential(BaseCompose):
    def __init__(self, transforms, record: bool = False, angle_version: str = 'oc'):
        super().__init__(transforms)
        self.record = record
        self.enable_record(record)
        self.angle_version = angle_version
        self.set_angle_version(angle_version)

    def enable_record(self, mode: bool = True):
        # enable children to record
        self.record = mode
        for transform in self.transforms:
            transform.enable_record(mode)
    
    def set_angle_version(self, version):
        self.angle_version = version
        for transform in self.transforms:
            transform.set_angle_version(version)
        


@PIPELINES.register_module()
class OneOf(Sequential):
    def __init__(self, transforms, record: bool = False, angle_version: str = 'oc'):
        self.transforms = []
        for trans in transforms:
            if isinstance(trans, list):
                self.transforms.append(Sequential(trans))
            else:
                assert isinstance(trans, dict)
                self.transforms.append(Sequential([trans]))
        self.enable_record(record)
        self.set_angle_version(angle_version)

    def __call__(self, results):
        transform = np.random.choice(self.transforms)
        return transform(results)


@PIPELINES.register_module()
class ShuffledSequential(Sequential):
    def __call__(self, data):
        order = np.random.permutation(len(self.transforms))
        for idx in order:
            t = self.transforms[idx]
            data = t(data)
            if data is None:
                return None
        return data


"""
Geometric Augmentation. Modified from https://github.com/microsoft/SoftTeacher/blob/main/ssod/datasets/pipelines/rand_aug.py
"""


def bbox2fields():
    """The key correspondence from bboxes to labels, masks and
    segmentations."""
    bbox2label = {"gt_bboxes": "gt_labels", "gt_bboxes_ignore": "gt_labels_ignore"}
    bbox2mask = {"gt_bboxes": "gt_masks", "gt_bboxes_ignore": "gt_masks_ignore"}
    bbox2seg = {
        "gt_bboxes": "gt_semantic_seg",
    }
    return bbox2label, bbox2mask, bbox2seg

class GeometricAugmentation(object):
    def __init__(
        self,
        img_fill_val=125,
        seg_ignore_label=255,
        min_size=0,
        prob: float = 1.0,
        random_magnitude: bool = True,
        record: bool = False,
        angle_version = 'oc',
    ):
        if isinstance(img_fill_val, (float, int)):
            img_fill_val = tuple([float(img_fill_val)] * 3)
        elif isinstance(img_fill_val, tuple):
            assert len(img_fill_val) == 3, "img_fill_val as tuple must have 3 elements."
            img_fill_val = tuple([float(val) for val in img_fill_val])
        assert np.all(
            [0 <= val <= 255 for val in img_fill_val]
        ), "all elements of img_fill_val should between range [0,255]."
        self.img_fill_val = img_fill_val
        self.seg_ignore_label = seg_ignore_label
        self.min_size = min_size
        self.prob = prob
        self.random_magnitude = random_magnitude
        self.record = record
        self.angle_version = angle_version

    def __call__(self, results):
        if np.random.random() < self.prob:
            magnitude: dict = self.get_magnitude(results)
            if self.record:
                if "aug_info" not in results:
                    results["aug_info"] = []
                results["aug_info"].append(self.get_aug_info(**magnitude))
            results = self.apply(results, **magnitude)
            self._filter_invalid(results, min_size=self.min_size)
        return results

    def get_magnitude(self, results) -> dict:
        raise NotImplementedError()

    def apply(self, results, **kwargs):
        raise NotImplementedError()

    def enable_record(self, mode: bool = True):
        self.record = mode
    
    def set_angle_version(self, version):
        self.angle_version = version

    def get_aug_info(self, **kwargs):
        aug_info = dict(type=self.__class__.__name__)
        aug_info.update(
            dict(
                # make op deterministic
                prob=1.0,
                random_magnitude=False,
                record=False,
                img_fill_val=self.img_fill_val,
                seg_ignore_label=self.seg_ignore_label,
                min_size=self.min_size,
            )
        )
        aug_info.update(kwargs)
        return aug_info

    def _filter_invalid(self, results, min_size=0):
        """Filter bboxes and masks too small or translated out of image.
        In Dota the dataformat is xywha, while in COCO it is xyxy, so I reformat that part.
        """
        if min_size is None:
            return results
        bbox2label, bbox2mask, _ = bbox2fields()
        for key in results.get("bbox_fields", []):
            bbox_w = results[key][:, 2] #- results[key][:, 0]
            bbox_h = results[key][:, 3] #- results[key][:, 1]
            valid_inds = (bbox_w > min_size) & (bbox_h > min_size)
            valid_inds = np.nonzero(valid_inds)[0]
            results[key] = results[key][valid_inds]
            # label fields. e.g. gt_labels and gt_labels_ignore
            label_key = bbox2label.get(key)
            if label_key in results:
                results[label_key] = results[label_key][valid_inds]
            # mask fields, e.g. gt_masks and gt_masks_ignore
            mask_key = bbox2mask.get(key)
            if mask_key in results:
                results[mask_key] = results[mask_key][valid_inds]
        return results

    def __repr__(self):
        return f"""{self.__class__.__name__}(
        img_fill_val={self.img_fill_val},
        seg_ignore_label={self.seg_ignore_label},
        min_size={self.magnitude},
        prob: float = {self.prob},
        random_magnitude: bool = {self.random_magnitude},
        )"""
############################################################################
@PIPELINES.register_module()
class RandTranslate(GeometricAugmentation):
    def __init__(self, x=None, y=None, **kwargs):
        super().__init__(**kwargs)
        self.x = x
        self.y = y
        if self.x is None and self.y is None:
            self.prob = 0.0

    def get_magnitude(self, results):
        magnitude = {}
        if self.random_magnitude:
            if isinstance(self.x, (list, tuple)):
                assert len(self.x) == 2
                x = np.random.random() * (self.x[1] - self.x[0]) + self.x[0]
                magnitude["x"] = x
            if isinstance(self.y, (list, tuple)):
                assert len(self.y) == 2
                y = np.random.random() * (self.y[1] - self.y[0]) + self.y[0]
                magnitude["y"] = y
        else:
            if self.x is not None:
                assert isinstance(self.x, (int, float))
                magnitude["x"] = self.x
            if self.y is not None:
                assert isinstance(self.y, (int, float))
                magnitude["y"] = self.y
        return magnitude

    def apply(self, results, x=None, y=None):
        # ratio to pixel
        h, w, c = results["img_shape"]
        if x is not None:
            x = w * x
        if y is not None:
            y = h * y
        if x is not None:
            # translate horizontally
            self._translate(results, x)
        if y is not None:
            # translate veritically
            self._translate(results, y, direction="vertical")
        return results

    def _translate(self, results, offset, direction="horizontal"):
        if self.record:
            GTrans.apply(
                results,
                "shift",
                dx=offset if direction == "horizontal" else 0,
                dy=offset if direction == "vertical" else 0,
            )
        self._translate_img(results, offset, direction=direction)
        self._translate_bboxes(results, offset, direction=direction)
        # fill_val defaultly 0 for BitmapMasks and None for PolygonMasks.
        self._translate_masks(results, offset, direction=direction)
        self._translate_seg(
            results, offset, fill_val=self.seg_ignore_label, direction=direction
        )

    def _translate_img(self, results, offset, direction="horizontal"):
        for key in results.get("img_fields", ["img"]):
            img = results[key].copy()
            results[key] = mmcv.imtranslate(
                img, offset, direction, self.img_fill_val
            ).astype(img.dtype)

    def _translate_bboxes(self, results, offset, direction="horizontal"):
        """Shift bboxes horizontally or vertically, according to offset."""
        h, w, c = results["img_shape"]
        for key in results.get("bbox_fields", []):
            # min_x, min_y, max_x, max_y = np.split(
            #     results[key], results[key].shape[-1], axis=-1
            # )
            # if direction == "horizontal":
            #     min_x = np.maximum(0, min_x + offset)
            #     max_x = np.minimum(w, max_x + offset)
            # elif direction == "vertical":
            #     min_y = np.maximum(0, min_y + offset)
            #     max_y = np.minimum(h, max_y + offset)

            # # the boxes translated outside of image will be filtered along with
            # # the corresponding masks, by invoking ``_filter_invalid``.
            # results[key] = np.concatenate([min_x, min_y, max_x, max_y], axis=-1)
            # Modification
            ########## second version #####################
            # x1, y1, x2, y2, x3, y3, x4, y4 = np.split(
            #     results[key], results[key].shape[-1], axis=-1
            # )
            # bbox_data = [x1, y1, x2, y2, x3, y3, x4, y4]
            
            # maxsize = w if direction == "horizontal" else h
            # for i in range(0,8,2):
            #     bbox_data[i] = np.maximum(0, bbox_data[i] + offset)
            #     bbox_data[i] = np.minimum(maxsize, bbox_data[i] + offset)
            

            # # the boxes translated outside of image will be filtered along with
            # # the corresponding masks, by invoking ``_filter_invalid``.
            # results[key] = np.concatenate([x1, y1, x2, y2, x3, y3, x4, y4], axis=-1)

            ########## Third version #####################
            box_xc, box_yc, box_w, box_h, box_a = np.split(
                results[key], results[key].shape[-1], axis=-1
            )
            if direction == "horizontal":
                box_xc = np.maximum(0, box_xc + offset)
                box_xc = np.minimum(w, box_xc + offset)
                # box_w = np.maximum(0, box_w + offset)
                # box_w = np.minimum(w, box_w + offset)
                
            elif direction == "vertical":
                box_yc = np.maximum(0, box_yc + offset)
                box_yc = np.minimum(h, box_yc + offset)
                # box_h = np.maximum(0, box_h + offset)
                # box_h = np.minimum(h, box_h + offset)
            
            # the boxes translated outside of image will be filtered along with
            # the corresponding masks, by invoking ``_filter_invalid``.
            results[key] = np.concatenate([box_xc, box_yc, box_w, box_h, box_a], axis=-1)
            

    def _translate_masks(self, results, offset, direction="horizontal", fill_val=0):
        """Translate masks horizontally or vertically."""
        h, w, c = results["img_shape"]
        for key in results.get("mask_fields", []):
            masks = results[key]
            results[key] = masks.translate((h, w), offset, direction, fill_val)

    def _translate_seg(self, results, offset, direction="horizontal", fill_val=255):
        """Translate segmentation maps horizontally or vertically."""
        for key in results.get("seg_fields", []):
            seg = results[key].copy()
            results[key] = mmcv.imtranslate(seg, offset, direction, fill_val).astype(
                seg.dtype
            )

    def __repr__(self):
        repr_str = super().__repr__()
        return ("\n").join(
            repr_str.split("\n")[:-1]
            + [f"x={self.x}", f"y={self.y}"]
            + repr_str.split("\n")[-1:]
        )



@PIPELINES.register_module()
class RandRotate(GeometricAugmentation):
    def __init__(self, angle=None, center=None, scale=1, **kwargs):
        super().__init__(**kwargs)
        self.angle = angle
        self.center = center
        self.scale = scale
        if self.angle is None:
            self.prob = 0.0

    def get_magnitude(self, results):
        magnitude = {}
        if self.random_magnitude:
            if isinstance(self.angle, (list, tuple)):
                assert len(self.angle) == 2
                angle = (
                    np.random.random() * (self.angle[1] - self.angle[0]) + self.angle[0]
                )
                magnitude["angle"] = angle
        else:
            if self.angle is not None:
                assert isinstance(self.angle, (int, float))
                magnitude["angle"] = self.angle

        return magnitude

    def apply(self, results, angle: float = None):
        h, w = results["img"].shape[:2]
        center = self.center
        if center is None:
            center = ((w - 1) * 0.5, (h - 1) * 0.5)
        self._rotate_img(results, angle, center, self.scale)
        rotate_matrix = cv2.getRotationMatrix2D(center, -angle, self.scale)
        if self.record:
            GTrans.apply(results, "rotate", cv2_rotation_matrix=rotate_matrix)
        self._rotate_bboxes(results, rotate_matrix)
        self._rotate_masks(results, angle, center, self.scale, fill_val=0)
        self._rotate_seg(
            results, angle, center, self.scale, fill_val=self.seg_ignore_label
        )
        return results

    def _rotate_img(self, results, angle, center=None, scale=1.0):
        """Rotate the image.
        Args:
            results (dict): Result dict from loading pipeline.
            angle (float): Rotation angle in degrees, positive values
                mean clockwise rotation. Same in ``mmcv.imrotate``.
            center (tuple[float], optional): Center point (w, h) of the
                rotation. Same in ``mmcv.imrotate``.
            scale (int | float): Isotropic scale factor. Same in
                ``mmcv.imrotate``.
        """
        for key in results.get("img_fields", ["img"]):
            img = results[key].copy()
            img_rotated = mmcv.imrotate(
                img, angle, center, scale, border_value=self.img_fill_val
            )
            results[key] = img_rotated.astype(img.dtype)

    def _rotate_bboxes(self, results, rotate_matrix):
        """Rotate the bboxes."""
        h, w, c = results["img_shape"]
        for key in results.get("bbox_fields", []):
            # min_x, min_y, max_x, max_y = np.split(
            #     results[key], results[key].shape[-1], axis=-1
            # )
            # coordinates = np.stack(
            #     [[min_x, min_y], [max_x, min_y], [min_x, max_y], [max_x, max_y]]
            # )  # [4, 2, nb_bbox, 1]
            # # pad 1 to convert from format [x, y] to homogeneous
            # # coordinates format [x, y, 1]
            # coordinates = np.concatenate(
            #     (
            #         coordinates,
            #         np.ones((4, 1, coordinates.shape[2], 1), coordinates.dtype),
            #     ),
            #     axis=1,
            # )  # [4, 3, nb_bbox, 1]
            # coordinates = coordinates.transpose((2, 0, 1, 3))  # [nb_bbox, 4, 3, 1]
            # rotated_coords = np.matmul(rotate_matrix, coordinates)  # [nb_bbox, 4, 2, 1]
            # rotated_coords = rotated_coords[..., 0]  # [nb_bbox, 4, 2]
            # min_x, min_y = (
            #     np.min(rotated_coords[:, :, 0], axis=1),
            #     np.min(rotated_coords[:, :, 1], axis=1),
            # )
            # max_x, max_y = (
            #     np.max(rotated_coords[:, :, 0], axis=1),
            #     np.max(rotated_coords[:, :, 1], axis=1),
            # )
            # min_x, min_y = (
            #     np.clip(min_x, a_min=0, a_max=w),
            #     np.clip(min_y, a_min=0, a_max=h),
            # )
            # max_x, max_y = (
            #     np.clip(max_x, a_min=min_x, a_max=w),
            #     np.clip(max_y, a_min=min_y, a_max=h),
            # )
            # results[key] = np.stack([min_x, min_y, max_x, max_y], axis=-1).astype(
            #     results[key].dtype
            # )

            ####################### Second Version ##############################
            gt_bboxes = np.concatenate(
                [results[key], np.zeros((results[key].shape[0], 1))], axis=-1)
            
            # polys = obb2poly_np(gt_bboxes, self.angle_version)[:, :-1]
            if gt_bboxes.shape[0] == 0:
                continue
            polys = obb2poly_np(gt_bboxes, self.angle_version)[:, :-1]
            x1, y1, x2, y2, x3, y3, x4, y4 = np.split(
                polys, polys.shape[-1], axis=-1
            )
            
            # x1, y1, x2, y2, x3, y3, x4, y4 = np.split(
            #     results[key], results[key].shape[-1], axis=-1
            # )
            coordinates = np.stack(
                [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            )  # [4, 2, nb_bbox, 1]
            # pad 1 to convert from format [x, y] to homogeneous
            # coordinates format [x, y, 1]
            coordinates = np.concatenate(
                (
                    coordinates,
                    np.ones((4, 1, coordinates.shape[2], 1), coordinates.dtype),
                ),
                axis=1,
            )  # [4, 3, nb_bbox, 1]
            coordinates = coordinates.transpose((2, 0, 1, 3))  # [nb_bbox, 4, 3, 1]
            rotated_coords = np.matmul(rotate_matrix, coordinates)  # [nb_bbox, 4, 2, 1]
            rotated_coords = rotated_coords.reshape((rotated_coords.shape[0],-1)) # [nb_bbox, 8]

            gt_bboxes = []
            for pt in rotated_coords:
                pt = np.array(pt, dtype=results[key].dtype)
                obb = poly2obb_np(pt, self.angle_version) \
                    if poly2obb_np(pt, self.angle_version) is not None\
                    else [0, 0, 0, 0, 0]
                gt_bboxes.append(obb)
            gt_bboxes = np.array(gt_bboxes, dtype=results[key].dtype)

            
            gt_bboxes[:,0] = np.clip(gt_bboxes[:,0], a_min=0, a_max=w)
            gt_bboxes[:,1] = np.clip(gt_bboxes[:,1], a_min=0, a_max=h)

            results[key] = gt_bboxes.astype(
                results[key].dtype
            )

            


    def _rotate_masks(self, results, angle, center=None, scale=1.0, fill_val=0):
        """Rotate the masks."""
        h, w, c = results["img_shape"]
        for key in results.get("mask_fields", []):
            masks = results[key]
            results[key] = masks.rotate((h, w), angle, center, scale, fill_val)

    def _rotate_seg(self, results, angle, center=None, scale=1.0, fill_val=255):
        """Rotate the segmentation map."""
        for key in results.get("seg_fields", []):
            seg = results[key].copy()
            results[key] = mmcv.imrotate(
                seg, angle, center, scale, border_value=fill_val
            ).astype(seg.dtype)

    def __repr__(self):
        repr_str = super().__repr__()
        return ("\n").join(
            repr_str.split("\n")[:-1]
            + [f"angle={self.angle}", f"center={self.center}", f"scale={self.scale}"]
            + repr_str.split("\n")[-1:]
        )


@PIPELINES.register_module()
class RandShear(GeometricAugmentation):
    def __init__(self, x=None, y=None, interpolation="bilinear", **kwargs):
        super().__init__(**kwargs)
        self.x = x
        self.y = y
        self.interpolation = interpolation
        if self.x is None and self.y is None:
            self.prob = 0.0

    def get_magnitude(self, results):
        magnitude = {}
        if self.random_magnitude:
            if isinstance(self.x, (list, tuple)):
                assert len(self.x) == 2
                x = np.random.random() * (self.x[1] - self.x[0]) + self.x[0]
                magnitude["x"] = x
            if isinstance(self.y, (list, tuple)):
                assert len(self.y) == 2
                y = np.random.random() * (self.y[1] - self.y[0]) + self.y[0]
                magnitude["y"] = y
        else:
            if self.x is not None:
                assert isinstance(self.x, (int, float))
                magnitude["x"] = self.x
            if self.y is not None:
                assert isinstance(self.y, (int, float))
                magnitude["y"] = self.y
        return magnitude

    def apply(self, results, x=None, y=None):
        if x is not None:
            # translate horizontally
            self._shear(results, np.tanh(-x * np.pi / 180))
        if y is not None:
            # translate veritically
            self._shear(results, np.tanh(y * np.pi / 180), direction="vertical")
        return results

    def _shear(self, results, magnitude, direction="horizontal"):
        if self.record:
            GTrans.apply(results, "shear", magnitude=magnitude, direction=direction)
        self._shear_img(results, magnitude, direction, interpolation=self.interpolation)
        self._shear_bboxes(results, magnitude, direction=direction)
        # fill_val defaultly 0 for BitmapMasks and None for PolygonMasks.
        self._shear_masks(
            results, magnitude, direction=direction, interpolation=self.interpolation
        )
        self._shear_seg(
            results,
            magnitude,
            direction=direction,
            interpolation=self.interpolation,
            fill_val=self.seg_ignore_label,
        )

    def _shear_img(
        self, results, magnitude, direction="horizontal", interpolation="bilinear"
    ):
        """Shear the image.
        Args:
            results (dict): Result dict from loading pipeline.
            magnitude (int | float): The magnitude used for shear.
            direction (str): The direction for shear, either "horizontal"
                or "vertical".
            interpolation (str): Same as in :func:`mmcv.imshear`.
        """
        for key in results.get("img_fields", ["img"]):
            img = results[key]
            img_sheared = mmcv.imshear(
                img,
                magnitude,
                direction,
                border_value=self.img_fill_val,
                interpolation=interpolation,
            )
            results[key] = img_sheared.astype(img.dtype)

    def _shear_bboxes(self, results, magnitude, direction="horizontal"):
        """Shear the bboxes."""
        h, w, c = results["img_shape"]
        if direction == "horizontal":
            shear_matrix = np.stack([[1, magnitude], [0, 1]]).astype(
                np.float32
            )  # [2, 2]
        else:
            shear_matrix = np.stack([[1, 0], [magnitude, 1]]).astype(np.float32)
        for key in results.get("bbox_fields", []):
            # min_x, min_y, max_x, max_y = np.split(
            #     results[key], results[key].shape[-1], axis=-1
            # )
            # coordinates = np.stack(
            #     [[min_x, min_y], [max_x, min_y], [min_x, max_y], [max_x, max_y]]
            # )  # [4, 2, nb_box, 1]
            # coordinates = (
            #     coordinates[..., 0].transpose((2, 1, 0)).astype(np.float32)
            # )  # [nb_box, 2, 4]
            # new_coords = np.matmul(
            #     shear_matrix[None, :, :], coordinates
            # )  # [nb_box, 2, 4]
            # min_x = np.min(new_coords[:, 0, :], axis=-1)
            # min_y = np.min(new_coords[:, 1, :], axis=-1)
            # max_x = np.max(new_coords[:, 0, :], axis=-1)
            # max_y = np.max(new_coords[:, 1, :], axis=-1)
            # min_x = np.clip(min_x, a_min=0, a_max=w)
            # min_y = np.clip(min_y, a_min=0, a_max=h)
            # max_x = np.clip(max_x, a_min=min_x, a_max=w)
            # max_y = np.clip(max_y, a_min=min_y, a_max=h)
            # results[key] = np.stack([min_x, min_y, max_x, max_y], axis=-1).astype(
            #     results[key].dtype
            # )
            ####################### Second Version ##############################
            gt_bboxes = np.concatenate(
                [results[key], np.zeros((results[key].shape[0], 1))], axis=-1)
            # polys = obb2poly_np(gt_bboxes, self.angle_version)[:, :-1]
            if gt_bboxes.shape[0] == 0:
                continue
            polys = obb2poly_np(gt_bboxes, self.angle_version)[:, :-1]
            x1, y1, x2, y2, x3, y3, x4, y4 = np.split(
                polys, polys.shape[-1], axis=-1
            )
            # x1, y1, x2, y2, x3, y3, x4, y4 = np.split(
            #     results[key], results[key].shape[-1], axis=-1
            # )
            coordinates = np.stack(
                [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            )  # [4, 2, nb_bbox, 1]
            coordinates = (
                coordinates[..., 0].transpose((2, 1, 0)).astype(np.float32)
            )  # [nb_box, 2, 4]
            new_coords = np.matmul(
                shear_matrix[None, :, :], coordinates
            )  # [nb_box, 2, 4]
            new_coords = new_coords.transpose((0, 2, 1)).reshape((new_coords.shape[0],-1)) # [nb_bbox, 8]

            gt_bboxes = []
            for pt in new_coords:
                pt = np.array(pt, dtype=results[key].dtype)
                obb = poly2obb_np(pt, self.angle_version) \
                    if poly2obb_np(pt, self.angle_version) is not None\
                    else [0, 0, 0, 0, 0]
                gt_bboxes.append(obb)
            gt_bboxes = np.array(gt_bboxes, dtype=results[key].dtype)

            
            gt_bboxes[:,0] = np.clip(gt_bboxes[:,0], a_min=0, a_max=w)
            gt_bboxes[:,1] = np.clip(gt_bboxes[:,1], a_min=0, a_max=h)

            results[key] = gt_bboxes.astype(
                results[key].dtype
            )


    def _shear_masks(
        self,
        results,
        magnitude,
        direction="horizontal",
        fill_val=0,
        interpolation="bilinear",
    ):
        """Shear the masks."""
        h, w, c = results["img_shape"]
        for key in results.get("mask_fields", []):
            masks = results[key]
            results[key] = masks.shear(
                (h, w),
                magnitude,
                direction,
                border_value=fill_val,
                interpolation=interpolation,
            )

    def _shear_seg(
        self,
        results,
        magnitude,
        direction="horizontal",
        fill_val=255,
        interpolation="bilinear",
    ):
        """Shear the segmentation maps."""
        for key in results.get("seg_fields", []):
            seg = results[key]
            results[key] = mmcv.imshear(
                seg,
                magnitude,
                direction,
                border_value=fill_val,
                interpolation=interpolation,
            ).astype(seg.dtype)

    def __repr__(self):
        repr_str = super().__repr__()
        return ("\n").join(
            repr_str.split("\n")[:-1]
            + [f"x_magnitude={self.x}", f"y_magnitude={self.y}"]
            + repr_str.split("\n")[-1:]
        )


@PIPELINES.register_module()
class RandErase(GeometricAugmentation):
    def __init__(
        self,
        n_iterations=None,
        size=None,
        squared: bool = True,
        patches=None,
        **kwargs,
    ):
        kwargs.update(min_size=None)
        super().__init__(**kwargs)
        self.n_iterations = n_iterations
        self.size = size
        self.squared = squared
        self.patches = patches

    def get_magnitude(self, results):
        magnitude = {}
        if self.random_magnitude:
            n_iterations = self._get_erase_cycle()
            patches = []
            h, w, c = results["img_shape"]
            for i in range(n_iterations):
                # random sample patch size in the image
                ph, pw = self._get_patch_size(h, w)
                # random sample patch left top in the image
                px, py = np.random.randint(0, w - pw), np.random.randint(0, h - ph)
                patches.append([px, py, px + pw, py + ph])
            magnitude["patches"] = patches
        else:
            assert self.patches is not None
            magnitude["patches"] = self.patches

        return magnitude

    def _get_erase_cycle(self):
        if isinstance(self.n_iterations, int):
            n_iterations = self.n_iterations
        else:
            assert (
                isinstance(self.n_iterations, (tuple, list))
                and len(self.n_iterations) == 2
            )
            n_iterations = np.random.randint(*self.n_iterations)
        return n_iterations

    def _get_patch_size(self, h, w):
        if isinstance(self.size, float):
            assert 0 < self.size < 1
            return int(self.size * h), int(self.size * w)
        else:
            assert isinstance(self.size, (tuple, list))
            assert len(self.size) == 2
            assert 0 <= self.size[0] < 1 and 0 <= self.size[1] < 1
            w_ratio = np.random.random() * (self.size[1] - self.size[0]) + self.size[0]
            h_ratio = w_ratio

            if not self.squared:
                h_ratio = (
                    np.random.random() * (self.size[1] - self.size[0]) + self.size[0]
                )
            return int(h_ratio * h), int(w_ratio * w)

    def apply(self, results, patches: list):
        for patch in patches:
            self._erase_image(results, patch, fill_val=self.img_fill_val)
            self._erase_mask(results, patch)
            self._erase_seg(results, patch, fill_val=self.seg_ignore_label)
        return results

    def _erase_image(self, results, patch, fill_val=128):
        for key in results.get("img_fields", ["img"]):
            tmp = results[key].copy()
            x1, y1, x2, y2 = patch
            tmp[y1:y2, x1:x2, :] = fill_val
            results[key] = tmp

    def _erase_mask(self, results, patch, fill_val=0):
        for key in results.get("mask_fields", []):
            masks = results[key]
            if isinstance(masks, PolygonMasks):
                # convert mask to bitmask
                masks = masks.to_bitmap()
            x1, y1, x2, y2 = patch
            tmp = masks.masks.copy()
            tmp[:, y1:y2, x1:x2] = fill_val
            masks = BitmapMasks(tmp, masks.height, masks.width)
            results[key] = masks

    def _erase_seg(self, results, patch, fill_val=0):
        for key in results.get("seg_fields", []):
            seg = results[key].copy()
            x1, y1, x2, y2 = patch
            seg[y1:y2, x1:x2] = fill_val
            results[key] = seg


@PIPELINES.register_module()
class RecomputeBox(object):
    def __init__(self, record=False,angle_version='oc'):
        self.record = record
        self.angle_version = angle_version
    def __call__(self, results):
        if self.record:
            if "aug_info" not in results:
                results["aug_info"] = []
            results["aug_info"].append(dict(type="RecomputeBox"))
        _, bbox2mask, _ = bbox2fields()
        for key in results.get("bbox_fields", []):
            mask_key = bbox2mask.get(key)
            if mask_key in results:
                masks = results[mask_key]
                results[key] = self._recompute_bbox(masks)
        return results

    def enable_record(self, mode: bool = True):
        self.record = mode

    def set_angle_version(self, version):
        self.angle_version = version

    def _recompute_bbox(self, masks):
        boxes = np.zeros(masks.masks.shape[0], 4, dtype=np.float32)
        x_any = np.any(masks.masks, axis=1)
        y_any = np.any(masks.masks, axis=2)
        for idx in range(masks.masks.shape[0]):
            x = np.where(x_any[idx, :])[0]
            y = np.where(y_any[idx, :])[0]
            if len(x) > 0 and len(y) > 0:
                boxes[idx, :] = np.array(
                    [x[0], y[0], x[-1] + 1, y[-1] + 1], dtype=np.float32
                )
        return boxes

###################### Inherit 'transforms' from 'mmrot' ######################
@PIPELINES.register_module()
class RandResize(rtrans.RResize):
    def __init__(self, record=False, **kwargs):
        super().__init__(**kwargs)
        self.record = record

    def __call__(self, results):
        results = super().__call__(results)
        if self.record:
            scale_factor = results["scale_factor"]
            GTrans.apply(results, "scale", sx=scale_factor[0], sy=scale_factor[1])

            if "aug_info" not in results:
                results["aug_info"] = []
            new_h, new_w = results["img"].shape[:2]
            results["aug_info"].append(
                dict(
                    type=self.__class__.__name__,
                    record=False,
                    img_scale=(new_w, new_h),
                    keep_ratio=False,
                    bbox_clip_border=self.bbox_clip_border,
                    backend=self.backend,
                )
            )
        return results

    def enable_record(self, mode: bool = True):
        self.record = mode
    
    def set_angle_version(self, version):
        self.angle_version = version


@PIPELINES.register_module()
class RRandFlip(rtrans.RRandomFlip):
    def __init__(self, record=False, **kwargs):
        super(RRandFlip,self).__init__(**kwargs)
        self.record = record

    def __call__(self, results):
        results = super().__call__(results)
        if self.record:
            if "aug_info" not in results:
                results["aug_info"] = []
            if results["flip"]:
                GTrans.apply(
                    results,
                    "flip",
                    direction=results["flip_direction"],
                    shape=results["img_shape"][:2],
                )
                results["aug_info"].append(
                    dict(
                        type=self.__class__.__name__,
                        record=False,
                        flip_ratio=1.0,
                        direction=results["flip_direction"],
                    )
                )
            else:
                results["aug_info"].append(
                    dict(
                        type=self.__class__.__name__,
                        record=False,
                        flip_ratio=0.0,
                        direction="vertical",
                    )
                )
        return results

    def enable_record(self, mode: bool = True):
        self.record = mode
    
    def set_angle_version(self, version):
        self.version = version
    

@PIPELINES.register_module()
class MultiBranch(object):
    def __init__(self, **transform_group):
        self.transform_group = {k: BaseCompose(v) for k, v in transform_group.items()}

    def __call__(self, results):
        multi_results = []
        for k, v in self.transform_group.items():
            res = v(copy.deepcopy(results))
            if res is None:
                return None
            # res["img_metas"]["tag"] = k
            multi_results.append(res)
        return multi_results

###################### Support YOLOX ######################
# code from https://github.com/liuyanyi/mmrotate/blob/ryolox/mmrotate/core/bbox/transforms.py
def find_inside_polygons(polygons, img_shape_x, img_shape_y):
    """Find inside polygons.

    Args:
        polygons (ndarray): Input polygons with shape (N,8).
        img_shape_x (int): Image shape x.
        img_shape_y (int): Image shape y.

    Returns:
        inside_ind (ndarray): Keep indices.
    """
    polygons_ctr_x = polygons[:, ::2].sum(axis=1) / 4
    polygons_ctr_y = polygons[:, 1::2].sum(axis=1) / 4
    polygons_inside_x = (polygons_ctr_x > 0) & (polygons_ctr_x < img_shape_x)
    polygons_inside_y = (polygons_ctr_y > 0) & (polygons_ctr_y < img_shape_y)
    # inside_ind = np.nonzero(polygons_inside_x & polygons_inside_y)[0]
    return polygons_inside_x & polygons_inside_y
 
@PIPELINES.register_module()
class PolyMixUp(transforms.MixUp):
    """Polygon MixUp data augmentation.

    .. code:: text

                         mixup transform
                +------------------------------+
                | mixup image   |              |
                |      +--------|--------+     |
                |      |        |        |     |
                |---------------+        |     |
                |      |                 |     |                     
                |      |      image      |     |
                |      |                 |     |
                |      |                 |     |
                |      |-----------------+     |
                |             pad              |
                +------------------------------+

     The mixup transform steps are as follows:

        1. Another random image is picked by dataset and embedded in
           the top left patch(after padding and resizing)
        2. The target of mixup transform is the weighted average of mixup
           image and origin image.

    Args:
        version (str, optional): Angle representations. Defaults to 'oc'.
        img_scale (Sequence[int]): Image output size after mixup pipeline.
            The shape order should be (height, width). Default: (640, 640).
        ratio_range (Sequence[float]): Scale ratio of mixup image.
            Default: (0.5, 1.5).
        flip_ratio (float): Horizontal flip ratio of mixup image.
            Default: 0.5.
        pad_val (int): Pad value. Default: 114.
        max_iters (int): The maximum number of iterations. If the number of
            iterations is greater than `max_iters`, but gt_bbox is still
            empty, then the iteration is terminated. Default: 15.
        min_bbox_size (float): Width and height threshold to filter bboxes.
            If the height or width of a box is smaller than this value, it
            will be removed. Default: 5.
        min_area_ratio (float): Threshold of area ratio between
            original bboxes and wrapped bboxes. If smaller than this value,
            the box will be removed. Default: 0.2.
        max_aspect_ratio (float): Aspect ratio of width and height
            threshold to filter bboxes. If max(h/w, w/h) larger than this
            value, the box will be removed. Default: 20.
        bbox_clip_border (bool, optional): Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
        skip_filter (bool): Whether to skip filtering rules. If it
            is True, the filter rule will not be applied, and the
            `min_bbox_size` and `min_area_ratio` and `max_aspect_ratio`
            is invalid. Default to True.
    """

    def __init__(self, version='oc', **kwargs):
        super(PolyMixUp, self).__init__(**kwargs)
        self.angle_version = version

    def _mixup_transform(self, results):
        """MixUp transform function.

        Args:
            results (dict): Result dict.
        Returns:
            dict: Updated result dict.
        """

        assert 'mix_results' in results
        assert len(
            results['mix_results']) == 1, 'MixUp only support 2 images now !'

        if results['mix_results'][0]['gt_bboxes'].shape[0] == 0:
            # empty bbox
            return results

        if len(results['gt_bboxes']) == 0:
            gt_polygons = np.zeros((0, 8))
        else:
            scores = np.zeros([len(results['gt_bboxes']), 1])
            bboxes_ws = np.concatenate([results['gt_bboxes'], scores], axis=1)
            gt_polygons = obb2poly_np(bboxes_ws, self.angle_version)
            gt_polygons = gt_polygons[:, :8]

        retrieve_results = results['mix_results'][0].copy()
        retrieve_img = retrieve_results['img'].copy()
        # show_img(retrieve_img, retrieve_results['gt_polygons'])

        jit_factor = random.uniform(*self.ratio_range)
        is_filp = random.uniform(0, 1) > self.flip_ratio

        if len(retrieve_img.shape) == 3:
            out_img = np.ones(
                (self.dynamic_scale[0], self.dynamic_scale[1], 3),
                dtype=retrieve_img.dtype) * self.pad_val
        else:
            out_img = np.ones(
                self.dynamic_scale, dtype=retrieve_img.dtype) * self.pad_val

        # 1. keep_ratio resize
        scale_ratio = min(self.dynamic_scale[0] / retrieve_img.shape[0],
                          self.dynamic_scale[1] / retrieve_img.shape[1])
        retrieve_img = mmcv.imresize(
            retrieve_img, (int(retrieve_img.shape[1] * scale_ratio),
                           int(retrieve_img.shape[0] * scale_ratio)))

        # 2. paste
        out_img[:retrieve_img.shape[0], :retrieve_img.shape[1]] = retrieve_img

        # 3. scale jit
        scale_ratio *= jit_factor
        out_img = mmcv.imresize(out_img, (int(out_img.shape[1] * jit_factor),
                                          int(out_img.shape[0] * jit_factor)))

        # 4. flip
        if is_filp:
            out_img = out_img[:, ::-1, :]

        # 5. random crop
        ori_img = results['img']
        origin_h, origin_w = out_img.shape[:2]
        target_h, target_w = ori_img.shape[:2]
        padded_img = np.zeros(
            (max(origin_h, target_h), max(origin_w,
                                          target_w), 3)).astype(np.uint8)
        padded_img[:origin_h, :origin_w] = out_img

        x_offset, y_offset = 0, 0
        if padded_img.shape[0] > target_h:
            y_offset = random.randint(0, padded_img.shape[0] - target_h)
        if padded_img.shape[1] > target_w:
            x_offset = random.randint(0, padded_img.shape[1] - target_w)
        padded_cropped_img = padded_img[y_offset:y_offset + target_h,
                                        x_offset:x_offset + target_w]

        # 6. adjust polygon
        retrieve_gt_bboxes = retrieve_results['gt_bboxes']
        # obb to polygon
        scores = np.zeros([len(retrieve_gt_bboxes), 1])
        bboxes_ws = np.concatenate([retrieve_gt_bboxes, scores], axis=1)
        polygons = obb2poly_np(bboxes_ws, self.angle_version)
        retrieve_gt_polygons = polygons[:, :8]

        retrieve_gt_polygons[:,
                             0::2] = retrieve_gt_polygons[:,
                                                          0::2] * scale_ratio
        retrieve_gt_polygons[:,
                             1::2] = retrieve_gt_polygons[:,
                                                          1::2] * scale_ratio
        if self.bbox_clip_border:
            retrieve_gt_polygons[:,
                                 0::2] = np.clip(retrieve_gt_polygons[:, 0::2],
                                                 0, origin_w)
            retrieve_gt_polygons[:,
                                 1::2] = np.clip(retrieve_gt_polygons[:, 1::2],
                                                 0, origin_h)

        if is_filp:
            retrieve_gt_polygons[:, 0::2] = (
                origin_w - retrieve_gt_polygons[:, 0::2])
            score = np.zeros((retrieve_gt_polygons.shape[0], 1))
            retrieve_gt_polygons = np.concatenate(
                [retrieve_gt_polygons, score], axis=1)
            retrieve_gt_polygons = get_best_begin_point(retrieve_gt_polygons)
            retrieve_gt_polygons = retrieve_gt_polygons[:, :8]

        # 7. filter
        cp_retrieve_gt_polygons = retrieve_gt_polygons.copy()
        cp_retrieve_gt_polygons[:, 0::2] = \
            cp_retrieve_gt_polygons[:, 0::2] - x_offset
        cp_retrieve_gt_polygons[:, 1::2] = \
            cp_retrieve_gt_polygons[:, 1::2] - y_offset

        # 8. mix up
        ori_img = ori_img.astype(np.float32)
        mixup_img = 0.5 * ori_img + 0.5 * padded_cropped_img.astype(np.float32)

        retrieve_gt_labels = retrieve_results['gt_labels']
        if not self.skip_filter:
            keep_list = self._filter_polygon_candidates(
                retrieve_gt_polygons.T, cp_retrieve_gt_polygons.T)

            retrieve_gt_labels = retrieve_gt_labels[keep_list]
            cp_retrieve_gt_polygons = cp_retrieve_gt_polygons[keep_list]

        mixup_gt_polygons = np.concatenate(
            (gt_polygons, cp_retrieve_gt_polygons), axis=0)
        mixup_gt_labels = np.concatenate(
            (results['gt_labels'], retrieve_gt_labels), axis=0)

        # remove outside bbox
        inside_inds = find_inside_polygons(mixup_gt_polygons, target_h,
                                           target_w)
        mixup_gt_polygons = mixup_gt_polygons[inside_inds]
        mixup_gt_labels = mixup_gt_labels[inside_inds]

        if self.bbox_clip_border:
            mixup_gt_polygons[:, 0::2] = np.clip(mixup_gt_polygons[:, 0::2], 0,
                                                 target_w)
            mixup_gt_polygons[:, 1::2] = np.clip(mixup_gt_polygons[:, 1::2], 0,
                                                 target_h)

        # polygon to obb
        mixup_gt_bboxes = poly2obb(
            torch.Tensor(mixup_gt_polygons),
            version=self.angle_version).numpy()

        results['img'] = mixup_img.astype(np.uint8)
        results['img_shape'] = mixup_img.shape
        results['gt_bboxes'] = mixup_gt_bboxes
        results['gt_labels'] = mixup_gt_labels
        return results

    def _filter_polygon_candidates(self, poly1, poly2):
        """Compute candidate boxes which include following 5 things:

        bbox1 before augment, bbox2 after augment, min_bbox_size (pixels),
        min_area_ratio, max_aspect_ratio.
        """
        bbox1 = poly2obb(
            torch.Tensor(poly1), version=self.angle_version).numpy()
        bbox2 = poly2obb(
            torch.Tensor(poly2), version=self.angle_version).numpy()

        w1, h1 = bbox1[2], bbox1[3]
        w2, h2 = bbox2[2], bbox2[3]
        ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))
        return ((w2 > self.min_bbox_size)
                & (h2 > self.min_bbox_size)
                & (w2 * h2 / (w1 * h1 + 1e-16) > self.min_area_ratio)
                & (ar < self.max_aspect_ratio))

@PIPELINES.register_module()
class PolyRandomAffine(transforms.RandomAffine):
    """Poly Random affine transform data augmentation.

    This operation randomly generates affine transform matrix which including
    rotation, translation, shear and scaling transforms.

    Args:
        version (str, optional): Angle representations. Defaults to 'oc'.
        max_rotate_degree (float): Maximum degrees of rotation transform.
            Default: 10.
        max_translate_ratio (float): Maximum ratio of translation.
            Default: 0.1.
        scaling_ratio_range (tuple[float]): Min and max ratio of
            scaling transform. Default: (0.5, 1.5).
        max_shear_degree (float): Maximum degrees of shear
            transform. Default: 2.
        border (tuple[int]): Distance from height and width sides of input
            image to adjust output shape. Only used in mosaic dataset.
            Default: (0, 0).
        border_val (tuple[int]): Border padding values of 3 channels.
            Default: (114, 114, 114).
        min_bbox_size (float): Width and height threshold to filter bboxes.
            If the height or width of a box is smaller than this value, it
            will be removed. Default: 2.
        min_area_ratio (float): Threshold of area ratio between
            original bboxes and wrapped bboxes. If smaller than this value,
            the box will be removed. Default: 0.2.
        max_aspect_ratio (float): Aspect ratio of width and height
            threshold to filter bboxes. If max(h/w, w/h) larger than this
            value, the box will be removed.
        bbox_clip_border (bool, optional): Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
        skip_filter (bool): Whether to skip filtering rules. If it
            is True, the filter rule will not be applied, and the
            `min_bbox_size` and `min_area_ratio` and `max_aspect_ratio`
            is invalid. Default to True.
    """

    def __init__(self, version='oc', **kwargs):
        super(PolyRandomAffine, self).__init__(**kwargs)
        self.angle_version = version

    def __call__(self, results):
        img = results['img']
        height = img.shape[0] + self.border[0] * 2
        width = img.shape[1] + self.border[1] * 2

        # Rotation
        rotation_degree = random.uniform(-self.max_rotate_degree,
                                         self.max_rotate_degree)
        rotation_matrix = self._get_rotation_matrix(rotation_degree)

        # Scaling
        scaling_ratio = random.uniform(self.scaling_ratio_range[0],
                                       self.scaling_ratio_range[1])
        scaling_matrix = self._get_scaling_matrix(scaling_ratio)

        # Shear
        x_degree = random.uniform(-self.max_shear_degree,
                                  self.max_shear_degree)
        y_degree = random.uniform(-self.max_shear_degree,
                                  self.max_shear_degree)
        shear_matrix = self._get_shear_matrix(x_degree, y_degree)

        # Translation
        trans_x = random.uniform(-self.max_translate_ratio,
                                 self.max_translate_ratio) * width
        trans_y = random.uniform(-self.max_translate_ratio,
                                 self.max_translate_ratio) * height
        translate_matrix = self._get_translation_matrix(trans_x, trans_y)

        warp_matrix = (
            translate_matrix @ shear_matrix @ rotation_matrix @ scaling_matrix)

        img = cv2.warpPerspective(
            img,
            warp_matrix,
            dsize=(width, height),
            borderValue=self.border_val)
        results['img'] = img
        results['img_shape'] = img.shape

        for key in results.get('bbox_fields', []):
            bboxes = results[key]
            num_bboxes = len(bboxes)
            assert bboxes.shape[1] == 5, 'bbox dim should be 5'
            if num_bboxes:
                # obb to polygon
                scores = np.zeros([num_bboxes, 1])
                bboxes_ws = np.concatenate([bboxes, scores], axis=1)
                polygons = obb2poly_np(bboxes_ws, self.angle_version)
                polygons = polygons[:, :8]

                # homogeneous coordinates
                xs = polygons[:, ::2].reshape(num_bboxes * 4)
                ys = polygons[:, 1::2].reshape(num_bboxes * 4)
                ones = np.ones_like(xs)
                points = np.vstack([xs, ys, ones])

                warp_points = warp_matrix @ points
                warp_points = warp_points[:2] / warp_points[2]
                xs = warp_points[0].reshape(num_bboxes, 4)
                ys = warp_points[1].reshape(num_bboxes, 4)

                warp_polygons = np.zeros_like(polygons)
                warp_polygons[:, ::2] = xs
                warp_polygons[:, 1::2] = ys

                # remove outside bbox
                valid_index = find_inside_polygons(warp_polygons, height,
                                                   width)

                if self.bbox_clip_border:
                    warp_polygons[:, ::2] = \
                        warp_polygons[:, ::2].clip(0, width)
                    warp_polygons[:, 1::2] = \
                        warp_polygons[:, 1::2].clip(0, height)

                wrap_bboxes = poly2obb(
                    torch.Tensor(warp_polygons),
                    version=self.angle_version).numpy()

                if not self.skip_filter:
                    # filter rbboxes
                    resize_bbox = bboxes.copy()
                    resize_bbox[:, 0:4] = resize_bbox[:, 0:4] * scaling_ratio
                    filter_index = self.filter_gt_bboxes(
                        resize_bbox, wrap_bboxes)
                    valid_index = valid_index & filter_index

                results[key] = wrap_bboxes[valid_index]
                if key in ['gt_bboxes']:
                    if 'gt_labels' in results:
                        results['gt_labels'] = results['gt_labels'][
                            valid_index]

                if 'gt_masks' in results:
                    raise NotImplementedError(
                        'RRandomAffine only supports bbox.')

        return results

    def filter_gt_bboxes(self, origin_bboxes, wrapped_bboxes):
        origin_w = origin_bboxes[:, 2]
        origin_h = origin_bboxes[:, 3]
        wrapped_w = wrapped_bboxes[:, 2]
        wrapped_h = wrapped_bboxes[:, 3]
        aspect_ratio = np.maximum(wrapped_w / (wrapped_h + 1e-16),
                                  wrapped_h / (wrapped_w + 1e-16))

        wh_valid_idx = (wrapped_w > self.min_bbox_size) & \
                       (wrapped_h > self.min_bbox_size)
        area_valid_idx = wrapped_w * wrapped_h / (origin_w * origin_h +
                                                  1e-16) > self.min_area_ratio
        aspect_ratio_valid_idx = aspect_ratio < self.max_aspect_ratio
        return wh_valid_idx & area_valid_idx & aspect_ratio_valid_idx

