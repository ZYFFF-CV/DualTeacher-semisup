"""This script build a 'Sampler' for 'SemiDataset' in 'dataset_wrapers.py', 
which is inherited from 'ConcatDataset' 'mmdetection/mmdet/datasets/dataset_wrappers.py', inherited from Pytorch 'ConcatDataset'
some tutorial may help:

Introduction to group sampler:https://zhuanlan.zhihu.com/p/463662605
Intro. to softTeacher: https://zhuanlan.zhihu.com/p/437754834

Note: 
 1. assert len(indices) == len(self) 报错, 原因之一是Falg是全0, 经过修改后, 本项目中的1和0与数据对应
 但是此改动导致'shuffled_indice_per_dataset'一部分为0,基于此进行改正
"""
from __future__ import division

import numpy as np
import torch
from mmcv.runner import get_dist_info
from torch.utils.data import Sampler, WeightedRandomSampler

from ..builder import SAMPLERS

@SAMPLERS.register_module()
class DistributedGroupSemiBalanceSampler2(Sampler):
    def __init__(
        self,
        dataset,
        by_prob=False,
        epoch_length=7330,
        sample_ratio=None,
        samples_per_gpu=1,
        num_replicas=None,
        rank=None,
        **kwargs
    ):
        # check to avoid some problem
        assert samples_per_gpu > 1, "samples_per_gpu should be greater than 1."
        _rank, _num_replicas = get_dist_info() #rank: 当前进程编号，num_replicas: 进程数量，与分布式训练有关, 默认 为`world_size`
        if num_replicas is None:
            num_replicas = _num_replicas
        if rank is None:
            rank = _rank

        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.by_prob = by_prob

        assert hasattr(self.dataset, "flag")
        # flag标志由dataset在初始化时确定，详见customdataset
        # flag只有两个取值，0: 'sup' 1: 'unspup'
        self.flag = self.dataset.flag
        self.group_sizes = np.bincount(self.flag) # 对每组的数量进行计数，详见bincount的使用方法
        self.num_samples = 0
        self.cumulative_sizes = dataset.cumulative_sizes #list, recording the start id of each dataset, see doc in pytorch
        # decide the frequency to sample each kind of datasets
        if not isinstance(sample_ratio, list):
            sample_ratio = [sample_ratio] * len(self.cumulative_sizes)
        self.sample_ratio = sample_ratio #sample_ratio=[1, 4], in 'configs/soft_teacher/base.py' is a list of 'int'
        self.sample_ratio = [
            int(sr / min(self.sample_ratio)) for sr in self.sample_ratio
        ]
        self.size_of_dataset = []
        cumulative_sizes = [0] + self.cumulative_sizes

        for i, _ in enumerate(self.group_sizes):
            size_of_dataset = 0
            cur_group_inds = np.where(self.flag == i)[0]
            for j in range(len(self.cumulative_sizes)):
                cur_group_cur_dataset = np.where(
                    np.logical_and(
                        cur_group_inds > cumulative_sizes[j],
                        cur_group_inds < cumulative_sizes[j + 1],
                    )
                )[0]
                size_per_dataset = len(cur_group_cur_dataset)
                size_of_dataset = max(
                    size_of_dataset, np.ceil(size_per_dataset / self.sample_ratio[j])
                )#如果sample ratio<1, 则加倍

            self.size_of_dataset.append(
                int(np.ceil(size_of_dataset / self.samples_per_gpu / self.num_replicas))
                * self.samples_per_gpu
            )
            for j in range(len(self.cumulative_sizes)):
                self.num_samples += self.size_of_dataset[-1] * self.sample_ratio[j]

        self.total_size = self.num_samples * self.num_replicas
        group_factor = [g / sum(self.group_sizes) for g in self.group_sizes]
        self.epoch_length = [int(np.round(gf * epoch_length)) for gf in group_factor]
        self.epoch_length[-1] = epoch_length - sum(self.epoch_length[:-1])

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = []
        cumulative_sizes = [0] + self.cumulative_sizes

        indice_per_dataset = []
        ################### Get image ID of each dataset ###################
        for i, size in enumerate(self.group_sizes):
            if size > 0: 
                indice = np.where(self.flag == i)[0] #取得属于某一类数据集的样本编号
                assert len(indice) == size
                indice_per_dataset.append(indice)
                

        shuffled_indice_per_dataset = [
            s[list(torch.randperm(int(s.shape[0]), generator=g).numpy())]
            for s in indice_per_dataset
        ]
       
        # split into
        total_indice = []
        batch_idx = 0
        # pdb.set_trace()
        ############## Sample image from two datasets for each batch #############
        while batch_idx < sum(self.epoch_length) * self.num_replicas:
            ratio = [x / sum(self.sample_ratio) for x in self.sample_ratio]
            if self.by_prob:
                indicator = list(
                    WeightedRandomSampler(
                        ratio,
                        self.samples_per_gpu,
                        replacement=True,
                        generator=g,
                    )
                )
                unique, counts = np.unique(indicator, return_counts=True)
                ratio = [0] * len(shuffled_indice_per_dataset)
                for u, c in zip(unique, counts):
                    ratio[u] = c
                assert len(ratio) == 2, "Only two set is supported"
                #如果某一数据集没有采样到则+1
                if ratio[0] == 0:
                    ratio[0] = 1
                    ratio[1] -= 1
                elif ratio[1] == 0:
                    ratio[1] = 1
                    ratio[0] -= 1

                ratio = [r / sum(ratio) for r in ratio]

            # num of each dataset
            ratio = [int(r * self.samples_per_gpu) for r in ratio]

            ratio[-1] = self.samples_per_gpu - sum(ratio[:-1])
            selected = []
            # print(ratio)
            for j in range(len(shuffled_indice_per_dataset)):
                ############ If not enough, refill ############
                if len(shuffled_indice_per_dataset[j]) < ratio[j]:
                    shuffled_indice_per_dataset[j] = np.concatenate(
                        (
                            shuffled_indice_per_dataset[j],
                            indice_per_dataset[j][
                                list(
                                    torch.randperm(
                                        int(indice_per_dataset[j].shape[0]),
                                        generator=g,
                                    ).numpy()
                                )
                            ],
                        )
                    )

                selected.append(shuffled_indice_per_dataset[j][: ratio[j]])
                shuffled_indice_per_dataset[j] = shuffled_indice_per_dataset[j][
                    ratio[j] :
                ]
            selected = np.concatenate(selected)
            total_indice.append(selected)
            batch_idx += 1
            # print(self.size_of_dataset)
        indice = np.concatenate(total_indice)
        indices.append(indice)
        indices = np.concatenate(indices)  # k
        indices = [
            indices[j]
            for i in list(
                torch.randperm(
                    len(indices) // self.samples_per_gpu,
                    generator=g,
                )
            )
            for j in range(
                i * self.samples_per_gpu,
                (i + 1) * self.samples_per_gpu,
            )
        ]

        offset = len(self) * self.rank
        indices = indices[offset : offset + len(self)]
        assert len(indices) == len(self)
        return iter(indices)

    def __len__(self):
        return sum(self.epoch_length) * self.samples_per_gpu

    def set_epoch(self, epoch):
        self.epoch = epoch