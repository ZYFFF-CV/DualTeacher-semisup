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
        reinit_parts=None,
        upgrade_norm=False
    ):

        self.reinit_parts = reinit_parts
        self.upgrade_norm = upgrade_norm

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

        if self.upgrade_norm:
             for (src_name, src_buffer), (tgt_name, tgt_buffer) in zip(
                    model.student.named_buffers(), model.teacher_onestage.named_buffers()
                ):
                if "running_mean" in src_name or "running_var" in src_name:
                    tgt_buffer.data.mul_(momentum).add_(src_buffer.data, alpha=1 - momentum)


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

@HOOKS.register_module()
class MultiTeacherWeightUpdater_onestage(TeacherWeightUpdater_onestage):
    """Used in Multi teacher, upgrade multiple senior 'teacher_onestage' 
    weight from student
    Args:
        upgrade_norm: whether upgrade norm layer from student
        
    """
    def __init__(
        self,
        reinit_parts=None,
        upgrade_norm=False,
    ):
        super().__init__(reinit_parts, upgrade_norm)

    def before_run(self, runner):
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        for i in range(1,model.num_senior+1):
            assert hasattr(model, f"teacher_onestage_{i}")
        assert hasattr(model, "student")
    
    def before_epoch(self, runner):
        model = runner.model
        if is_module_wrapper(model):
            model = model.module

        if runner.epoch == model.softlearning_after_epoch:
            log_every_n("Clone all parameters of student to teacher...", level=logging.INFO)
            for i in range(1,model.num_senior+1):
                model_teacher_onestage =  getattr(model, f"teacher_onestage_{i}")
                self.momentum_update(model.student, model_teacher_onestage, 0)
            self.weight_reinit(model.student, self.reinit_parts)

    def momentum_update(self, model_student,model_teacher_onestage, momentum):
        for (src_name, src_parm), (tgt_name, tgt_parm) in zip(
            model_student.named_parameters(), model_teacher_onestage.named_parameters()
        ):
            tgt_parm.data.mul_(momentum).add_(src_parm.data, alpha=1 - momentum)

        if self.upgrade_norm:
             for (src_name, src_buffer), (tgt_name, tgt_buffer) in zip(
                    model_student.named_buffers(), model_teacher_onestage.named_buffers()
                ):
                if "running_mean" in src_name or "running_var" in src_name:
                    tgt_buffer.data.mul_(momentum).add_(src_buffer.data, alpha=1 - momentum)