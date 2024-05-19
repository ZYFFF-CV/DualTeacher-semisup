from mmcv.parallel import is_module_wrapper
from mmcv.runner.hooks import HOOKS, Hook
from bisect import bisect_right
from ..logger import log_every_n
import logging

@HOOKS.register_module()
class TeacherWeightUpdater_onestage(Hook):
    """Used in Dual teacher, upgrade 'teacher_onestage' 
    weight from student
    """
    def __init__(
        self,
        reinit_parts=None
        # momentum=0.999,
        # interval=1,
        # warm_up=100,
        # decay_intervals=None,
        # decay_factor=0.1,
    ):
        # assert momentum >= 0 and momentum <= 1
        # self.momentum = momentum
        # assert isinstance(interval, int) and interval > 0
        # self.warm_up = warm_up
        # self.interval = interval
        # assert isinstance(decay_intervals, list) or decay_intervals is None
        # self.decay_intervals = decay_intervals
        # self.decay_factor = decay_factor
        self.reinit_parts = reinit_parts

    def before_run(self, runner):
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        assert hasattr(model, "teacher_onestage")
        assert hasattr(model, "student")
        # only do it at initial stage
        # if runner.epoch > model.softlearning_after_epoch:
        #     log_every_n("Clone all parameters of student to teacher...")
        #     self.momentum_update(model, 0)

    def before_epoch(self, runner):
        model = runner.model
        if is_module_wrapper(model):
            model = model.module

        if runner.epoch == model.softlearning_after_epoch:
            log_every_n("Clone all parameters of student to teacher...", level=logging.INFO)
            self.momentum_update(model, 0)
            self.weight_reinit(model.student, self.reinit_parts)

    def momentum_update(self, model, momentum):
        for (src_name, src_parm), (tgt_name, tgt_parm) in zip(
            model.student.named_parameters(), model.teacher_onestage.named_parameters()
        ):
            tgt_parm.data.mul_(momentum).add_(src_parm.data, alpha=1 - momentum)


    def weight_reinit(self, model, init_parts=None):
        """
        Reinit weights
        """
        if init_parts is None:
            model.init_weights()
            return
        
        for part in init_parts:
            if not hasattr(model, part):
                raise TypeError("{} is not in the {}".format(part,
                                model.__class__.__name__))
            modelpart = getattr(model, part)
            modelpart.init_weights()

        