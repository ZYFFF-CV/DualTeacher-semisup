from mmdet.datasets import DATASETS, ConcatDataset 
from mmrotate.datasets import build_dataset
import numpy as np


@DATASETS.register_module()
class SemiDataset(ConcatDataset):
    """Wrapper for semisupervised od."""

    def __init__(self, sup: dict, unsup: dict, **kwargs):
        super().__init__([build_dataset(sup), build_dataset(unsup)], **kwargs)

        flag = []
        cumulative_sizes = [0] + self.cumulative_sizes
        for i in range(len(cumulative_sizes)-1):
            flag += [i] * (cumulative_sizes[i+1] - cumulative_sizes[i])
        self.flag = np.array(flag, dtype=np.uint8)    

    @property
    def sup(self):
        return self.datasets[0]

    @property
    def unsup(self):
        return self.datasets[1]