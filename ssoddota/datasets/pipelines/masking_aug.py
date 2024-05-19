"""
Augmentation Masked Image Modeling (MIM) method

"""

from mmdet.datasets import PIPELINES
import numpy as np

@PIPELINES.register_module()
class ImgMasking:
    """Uniformally mask pixels

    Args:
        ratio (float): Rate of masked regions. Default 0.5.
        fill_val (array): value used to replace the origianl pixel, default imagenet mean
    """

    def __init__(self, ratio=0.5, fill_val=[123.675, 116.28, 103.53]):#
        self.ratio = ratio
        self.fill_val = fill_val

    def __call__(self, results):
        h, w, c = results["img_shape"]
        numDiscard = int(h*w*self.ratio)
        img = results['img']

        # Note that simple reshape is will also affect the original data,
        # as the reshapped data changes

        pixels = img.copy().reshape([h*w, c]) 

        # Shuffle and select the discard pixels
        ids = np.arange(h*w)
        np.random.shuffle(ids)
        ids_discard = ids[:numDiscard]
        mask = np.ones(h*w,dtype=np.int8)
        mask[ids_discard] = 0
        pixels[ids_discard] = self.fill_val 
        results['masked_img'] = pixels.reshape(h, w, c)
        # results['mask'] = mask.reshape(h,w)  
        results['gt_masks'] = mask.reshape(h,w) # use 'gt_masks' key temporary
        
        return results