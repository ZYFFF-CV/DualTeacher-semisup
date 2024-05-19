from mmcv.parallel import is_module_wrapper
from mmcv.runner.hooks import HOOKS, Hook
from bisect import bisect_right
from ..logger import log_every_n


@HOOKS.register_module()
class TemperatureDecay(Hook):
    """defay factors of 'unsup cls teacehr' loss
    """
    def __init__(
        self,
        minval = 0.1,
        
    ):
        self.decay_factor = 0
        self.minval = minval
       
        
    
    def before_run(self, runner):
        model = runner.model
        if is_module_wrapper(model):
            model = model.module

        reduction_epoches = runner.max_epochs - model.softlearning_after_epoch
        self.decay_factor = (1 - self.minval) / reduction_epoches
        # only do it at initial stage
        if runner.epoch > model.softlearning_after_epoch:
            model.upsup_teacher_pred_weight = 1 - self.decay_factor * (runner.epoch - model.softlearning_after_epoch)

    def after_epoch(self, runner):
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        if runner.epoch > model.softlearning_after_epoch:
            model.upsup_teacher_pred_weight = 1 - self.decay_factor * (runner.epoch - model.softlearning_after_epoch)