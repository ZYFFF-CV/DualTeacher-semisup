"""Mergerd Softteacher and MMRotate
"""
from .train import get_root_logger, set_random_seed, train_detector
from .inference_mmrot import inference_detector_by_patches
__all__ = ["get_root_logger", "set_random_seed", "train_detector","inference_detector_by_patches"]