"""Update:
1. 'log_image_with_boxes' call 'imshow_det_rbboxes' in mmrot for result visuallization
"""

import logging
import os
import sys
from collections import Counter
from typing import Tuple

import mmcv
import numpy as np
import torch
from mmcv.runner.dist_utils import get_dist_info
from mmcv.utils import get_logger
from mmdet.core.visualization import imshow_det_bboxes
from mmrotate.core.visualization import imshow_det_rbboxes
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import pandas as pd


try:
    import wandb
except:
    wandb = None

_log_counter = Counter()


def get_root_logger(log_file=None, log_level=logging.INFO):
    """Get root logger.
    Args:
        log_file (str, optional): File path of log. Defaults to None.
        log_level (int, optional): The level of logger.
            Defaults to logging.INFO.
    Returns:
        :obj:`logging.Logger`: The obtained logger
    """
    logger = get_logger(name="mmdet.ssod", log_file=log_file, log_level=log_level)
    logger.propagate = False
    return logger


def _find_caller():
    frame = sys._getframe(2)
    while frame:
        code = frame.f_code
        if os.path.join("utils", "logger.") not in code.co_filename:
            mod_name = frame.f_globals["__name__"]
            if mod_name == "__main__":
                mod_name = r"ssod"
            return mod_name, (code.co_filename, frame.f_lineno, code.co_name)
        frame = frame.f_back



def convert_box(tag, boxes, box_labels, class_labels, std, scores=None):
    if isinstance(std, int):
        std = [std, std]
    if len(std) != 4:
        std = std[::-1] * 2
    std = boxes.new_tensor(std).reshape(1, 4)
    wandb_box = {}
    boxes = boxes / std
    boxes = boxes.detach().cpu().numpy().tolist()
    box_labels = box_labels.detach().cpu().numpy().tolist()
    class_labels = {k: class_labels[k] for k in range(len(class_labels))}
    wandb_box["class_labels"] = class_labels
    assert len(boxes) == len(box_labels)
    if scores is not None:
        scores = scores.detach().cpu().numpy().tolist()
        box_data = [
            dict(
                position=dict(minX=box[0], minY=box[1], maxX=box[2], maxY=box[3]),
                class_id=label,
                scores=dict(cls=scores[i]),
            )
            for i, (box, label) in enumerate(zip(boxes, box_labels))
        ]
    else:
        box_data = [
            dict(
                position=dict(minX=box[0], minY=box[1], maxX=box[2], maxY=box[3]),
                class_id=label,
            )
            for i, (box, label) in enumerate(zip(boxes, box_labels))
        ]

    wandb_box["box_data"] = box_data
    return {tag: wandb.data_types.BoundingBoxes2D(wandb_box, tag)}


def color_transform(img_tensor, mean, std, to_rgb=False):
    img_np = img_tensor.detach().cpu().numpy().transpose((1, 2, 0)).astype(np.float32)
    return mmcv.imdenormalize(img_np, mean, std, to_bgr=not to_rgb)


def log_image_with_boxes(
    tag: str,
    image: torch.Tensor,
    bboxes: torch.Tensor,
    bbox_tag: str = None,
    labels: torch.Tensor = None,
    scores: torch.Tensor = None,
    class_names: Tuple[str] = None,
    filename: str = None,
    img_norm_cfg: dict = None,
    backend: str = "auto",
    interval: int = 50,
):
    rank, _ = get_dist_info()
    if rank != 0:
        return
    _, key = _find_caller()
    _log_counter[key] += 1
    if not (interval == 1 or _log_counter[key] % interval == 1):
        return
    if backend == "auto":
        if (wandb is None) or (wandb.run is None):
            backend = "file"
        else:
            backend = "wandb"

    if backend == "wandb":
        if wandb is None:
            raise ImportError("wandb is not installed")
        assert (
            wandb.run is not None
        ), "wandb has not been initialized, call `wandb.init` first`"

    elif backend != "file":
        raise TypeError("backend must be file or wandb")

    if filename is None:
        filename = f"{_log_counter[key]}.jpg"
    if bbox_tag is not None:
        bbox_tag = "vis"
    if img_norm_cfg is not None:
        image = color_transform(image, **img_norm_cfg)
    if labels is None:
        labels = bboxes.new_zeros(bboxes.shape[0]).long()
        class_names = ["foreground"]
    if backend == "wandb":
        im = {}
        im["data_or_path"] = image
        im["boxes"] = convert_box(
            bbox_tag, bboxes, labels, class_names, scores=scores, std=image.shape[:2]
        )
        wandb.log({tag: wandb.Image(**im)}, commit=False)
    elif backend == "file":
        root_dir = os.environ.get("WORK_DIR", ".")

        imshow_det_rbboxes(
            image,
            bboxes.cpu().detach().numpy(),
            labels.cpu().detach().numpy(),
            class_names=class_names,
            show=False,
            out_file=os.path.join(root_dir, tag, bbox_tag, filename),
        )
        # imshow_det_bboxes(
        #     image,
        #     bboxes.cpu().detach().numpy(),
        #     labels.cpu().detach().numpy(),
        #     class_names=class_names,
        #     show=False,
        #     out_file=os.path.join(root_dir, tag, bbox_tag, filename),
        # )
    else:
        raise TypeError("backend must be file or wandb")


def log_every_n(msg: str, n: int = 50, level: int = logging.DEBUG, backend="auto"):
    """
    Args:
        msg (Any):
        n (int):
        level (int):
        name (str):
    """
    caller_module, key = _find_caller()
    _log_counter[key] += 1
    if n == 1 or _log_counter[key] % n == 1:
        if isinstance(msg, dict) and (wandb is not None) and (wandb.run is not None):
            wandb.log(msg, commit=False)
        else:
            get_root_logger().log(level, msg)


def log_rpn_loss_kde(
 
    scores: torch.Tensor = None,
    labels: torch.Tensor = None,
    calsses: list = ['foreground','background'],
    filename: str = None,
    tag: str = "RPN_pn_kde",
    interval: int = 50,
):
    """Plot ked of histogram
    """
    rank, _ = get_dist_info()
    if rank != 0:
        return
    _, key = _find_caller()
    _log_counter[key] += 1
    if not (interval == 1 or _log_counter[key] % interval == 1):
        return
    ###### Only support 'file' backend

    if filename is None:
        filename = f"{_log_counter[key]}.png"
        
    root_dir = os.environ.get("WORK_DIR", ".")
    # print(scores.shape,labels.shape)
    # print("*"*10)
    # assert 0
    # plot kde
   
    data = {
    'value': scores[:,0].clone().detach().sigmoid().cpu().numpy(),
    'label': [calsses[i] for i in labels]
    }
    df = pd.DataFrame(data)

    
    # Plot kde
    # sns.kdeplot(data=df[df['label'] == 'foreground'], x='value', color='orange', label='foreground')
    # sns.kdeplot(data=df[df['label'] == 'background'], x='value', color='blue', label='background')
    # plt.title('KDE of foreground and background')

    # Plot hist
    # sns.histplot(data=df, x='value', hue='label', palette={'foreground': 'orange', 'background': 'blue'}, kde=False)
    # sns.histplot(data=df, x='value', hue='label', 
    #          palette={'foreground': 'orange', 'background': 'blue'},
    #          kde=False,
    #          multiple='layer', element='poly', common_norm=False)

    sns.histplot(data=df[df['label'] == 'background'], x='value', color='blue', 
                 element='poly', stat='count', label='background', common_norm=False)
    sns.histplot(data=df[df['label'] == 'foreground'], x='value', color='orange', 
                 element='poly', stat='count', label='foreground', common_norm=False)
    
    plt.title('Histogram of foreground and background')
    # plt.legend(title='Label', labels=['foreground', 'background'])
    
    plt.legend(title='Label')
    save_path = os.path.join(root_dir, tag, filename)
    if not os.path.exists(os.path.join(root_dir, tag)):
        os.mkdir(os.path.join(root_dir, tag))
    plt.savefig(save_path, dpi=600)
    plt.close()

    # else:
    #     raise TypeError("backend must be file or wandb")

def log_multicalss_loss_kde_fgbg(
 
    scores: torch.Tensor = None,
    labels: torch.Tensor = None,
    calsses: list = ['foreground','background'],
    filename: str = None,
    activation_func = None,
    tag: str = "RPN_pn_kde",
    interval: int = 50,
):
    """Plot kde of histogram for multicalss prediction,
    with only foregrtound and background.
   
    Args:
        scores: 2D torch tensor, torch.Size([N, num_calss]) 
        labels: 1D torch tensor, torch.Size([N])
    """
    rank, _ = get_dist_info()
    if rank != 0:
        return
    _, key = _find_caller()
    _log_counter[key] += 1
    if not (interval == 1 or _log_counter[key] % interval == 1):
        return
    ###### Only support 'file' backend

    if filename is None:
        filename = f"{_log_counter[key]}.png"
        
    root_dir = os.environ.get("WORK_DIR", ".")
    # plot kde
    bg_class_ind = scores.size(-1)
    scores_ = scores.clone().detach().cpu()
    fgbglabels = ((labels >= 0) & (labels < bg_class_ind)).cpu().numpy().astype(np.int)
    if activation_func == 'sigmoid':
        scores_activated = np.max(scores_.sigmoid().numpy(),-1)
    else:
        raise NotImplementedError("{} not implemented!".format(activation_func))

    # print(scores_) 
    # print(fgbglabels)  

    data = {
    'value': scores_activated,
    'label': [calsses[i] for i in fgbglabels]
    }
    df = pd.DataFrame(data)

    # print(scores_activated.shape)
    # print('*'*50)
    # Plot kde
    # sns.kdeplot(data=df[df['label'] == 'foreground'], x='value', color='orange', label='foreground')
    # sns.kdeplot(data=df[df['label'] == 'background'], x='value', color='blue', label='background')
    # plt.title('KDE of foreground and background')

    # Plot hist
    # sns.histplot(data=df, x='value', hue='label', palette={'foreground': 'orange', 'background': 'blue'}, kde=False)
    # sns.histplot(data=df, x='value', hue='label', 
    #          palette={'foreground': 'orange', 'background': 'blue'},
    #          kde=False,
    #          multiple='layer', element='poly', common_norm=False)

    sns.histplot(data=df[df['label'] == 'background'], x='value', color='blue', 
                 element='poly', stat='count', label='background', common_norm=False)
    sns.histplot(data=df[df['label'] == 'foreground'], x='value', color='orange', 
                 element='poly', stat='count', label='foreground', common_norm=False)
    
    plt.title('Histogram of foreground and background')
    # plt.legend(title='Label', labels=['foreground', 'background'])
    
    plt.legend(title='Label')
    save_path = os.path.join(root_dir, tag, filename)
    if not os.path.exists(os.path.join(root_dir, tag)):
        os.mkdir(os.path.join(root_dir, tag))
    plt.savefig(save_path, dpi=600)
    plt.close()
