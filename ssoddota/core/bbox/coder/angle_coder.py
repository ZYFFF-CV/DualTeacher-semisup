# copied from https://github.com/liuyanyi/mmrotate/blob/ryolox/mmrotate/core/bbox/coder/angle_coder.py

from mmdet.core.bbox.coder.base_bbox_coder import BaseBBoxCoder

from mmrotate.core.bbox.builder import ROTATED_BBOX_CODERS


@ROTATED_BBOX_CODERS.register_module()
class PseudoAngleCoder(BaseBBoxCoder):

    def __init__(self):
        super().__init__()
        self.coding_len = 1

    def encode(self, angle_targets):
        return angle_targets

    def decode(self, angle_preds):
        return angle_preds.squeeze(-1)