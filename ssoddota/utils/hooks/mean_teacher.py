from mmcv.parallel import is_module_wrapper
from mmcv.runner.hooks import HOOKS, Hook
from bisect import bisect_right
from ..logger import log_every_n


@HOOKS.register_module()
class MeanTeacher(Hook):
    def __init__(
        self,
        momentum=0.999,
        interval=1,
        warm_up=100,
        decay_intervals=None,
        decay_factor=0.1,

    ):
        assert momentum >= 0 and momentum <= 1
        self.momentum = momentum
        assert isinstance(interval, int) and interval > 0
        self.warm_up = warm_up
        self.interval = interval
        assert isinstance(decay_intervals, list) or decay_intervals is None
        self.decay_intervals = decay_intervals
        self.decay_factor = decay_factor

    def before_run(self, runner):
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        assert hasattr(model, "teacher")
        assert hasattr(model, "student")
        # only do it at initial stage
        if runner.iter == 0:
            log_every_n("Clone all parameters of student to teacher...")
            self.momentum_update(model, 0)

    def before_train_iter(self, runner):
        """Update ema parameter every self.interval iterations."""
        curr_step = runner.iter
        if curr_step % self.interval != 0:
            return
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        # We warm up the momentum considering the instability at beginning
        momentum = min(
            self.momentum, 1 - (1 + self.warm_up) / (curr_step + 1 + self.warm_up)
        )
        runner.log_buffer.output["ema_momentum"] = momentum
        self.momentum_update(model, momentum)

    def after_train_iter(self, runner):
        curr_step = runner.iter
        if self.decay_intervals is None:
            return
        self.momentum = 1 - (1 - self.momentum) * self.decay_factor ** bisect_right(
            self.decay_intervals, curr_step
        )

    def momentum_update(self, model, momentum):
        for (src_name, src_parm), (tgt_name, tgt_parm) in zip(
            model.student.named_parameters(), model.teacher.named_parameters()
        ):
            tgt_parm.data.mul_(momentum).add_(src_parm.data, alpha=1 - momentum)


@HOOKS.register_module()
class MeanTeacher_DualTeacehr(Hook):
    def __init__(
        self,
        momentum=0.999,
        interval=1,
        warm_up=100,
        decay_intervals=None,
        decay_factor=0.1,
        upgrade_norm=False
    ):
        assert momentum >= 0 and momentum <= 1
        self.momentum = momentum
        assert isinstance(interval, int) and interval > 0
        self.warm_up = warm_up
        self.interval = interval
        assert isinstance(decay_intervals, list) or decay_intervals is None
        self.decay_intervals = decay_intervals
        self.decay_factor = decay_factor

        self.unupdated_iter = 0
        self.upgrade_norm = upgrade_norm

    def before_run(self, runner):
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        assert hasattr(model, "teacher_onestage")
        assert hasattr(model, "student")
        # only do it at initial stage
        # if runner.iter == 0:
        #     log_every_n("Clone all parameters of student to teacher...")
        #     self.momentum_update(model, 0)

    def before_train_iter(self, runner):
        """Update ema parameter every self.interval iterations, 
        after warm up epoches."""
        model = runner.model
        if is_module_wrapper(model):
            model = model.module

        if not runner.epoch > model.softlearning_after_epoch: 
            self.unupdated_iter = runner.iter
            return
        
        curr_step = runner.iter
        if curr_step % self.interval != 0:
            return
        
        # We warm up the momentum considering the instability at beginning
        momentum = min(
            self.momentum, 1 - (1 + self.warm_up) / (curr_step - self.unupdated_iter + 1 + self.warm_up)
        )
        runner.log_buffer.output["ema_momentum"] = momentum
        self.momentum_update(model, momentum)

    def after_train_iter(self, runner):
        curr_step = runner.iter
        if self.decay_intervals is None:
            return
        self.momentum = 1 - (1 - self.momentum) * self.decay_factor ** bisect_right(
            self.decay_intervals, curr_step
        )

    def momentum_update(self, model, momentum):
        for (src_name, src_parm), (tgt_name, tgt_parm) in zip(
            model.student.named_parameters(), model.teacher_onestage.named_parameters()
        ):
            tgt_parm.data.mul_(momentum).add_(src_parm.data, alpha=1 - momentum)

        # Upgrade running mean and running variance in the Normalizaiton layers, such as YOLO, where the BN
        # is trained from scratch
        if self.upgrade_norm:
            for (src_name, src_buffer), (tgt_name, tgt_buffer) in zip(
                    model.student.named_buffers(), model.teacher_onestage.named_buffers()
                ):
                if "running_mean" in src_name or "running_var" in src_name:
                    tgt_buffer.data.mul_(momentum).add_(src_buffer.data, alpha=1 - momentum)
                    
@HOOKS.register_module()
class MeanTeacher_DualTeacher_epoch(Hook):
    """Exponential Moving Average Hook. Used in YOLO based semi-supervised learning 
    We do not hope the teacher models involed into the EMA update.
    Student model used only (or the specified module)
    Also the senior model is updated after several epochs rather than in iteration 
    as previous

   

    Args:
        momentum (float): The momentum used for updating ema parameter.
            Ema's parameter are updated with the formula:
           `ema_param = (1-momentum) * ema_param + momentum * cur_param`.
            Defaults to 0.0002.
      
        interval (int): Update ema parameter every interval epoch.
            Defaults to 1.
        
        upgrade_norm (bool): Whether allow normalization layer into updating, default False
    """

    def __init__(
        self,
        momentum=0.999,
        interval=1,
        # warm_up=100,
        decay_intervals=None,
        decay_factor=0.1,
        upgrade_norm=False
    ):
        assert momentum >= 0 and momentum <= 1
        self.momentum = momentum
        assert isinstance(interval, int) and interval > 0
        # self.warm_up = warm_up
        self.interval = interval
        assert isinstance(decay_intervals, list) or decay_intervals is None
        self.decay_intervals = decay_intervals
        self.decay_factor = decay_factor

        # self.unupdated_iter = 0
        self.upgrade_norm = upgrade_norm

    def before_run(self, runner):
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        assert hasattr(model, "teacher_onestage")
        assert hasattr(model, "student")
    
    def before_train_epoch(self, runner):
        """We recover model's parameter from ema backup after last epoch's
        EvalHook."""
        model = runner.model
        if is_module_wrapper(model):
            model = model.module

        if not runner.epoch > model.softlearning_after_epoch: 
            # self.unupdated_iter = runner.iter
            return
        
        # curr_epoch = runner.iter
        if runner.epoch % self.interval != 0:
            return
        
        # We warm up the momentum considering the instability at beginning
        # momentum = min(
        #     self.momentum, 1 - (1 + self.warm_up) / (curr_step - self.unupdated_iter + 1 + self.warm_up)
        # )
        # runner.log_buffer.output["ema_momentum"] = momentum
        # self.momentum_update(model, momentum)
        self.momentum_update(model, self.momentum)
    
    def after_train_epoch(self, runner):
        # curr_step = runner.iter
        curr_step = runner.epoch
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        self.momentum_update(model, self.momentum)
        if self.decay_intervals is None:
            return
        self.momentum = 1 - (1 - self.momentum) * self.decay_factor ** bisect_right(
            self.decay_intervals, curr_step
        )
        

    
    def momentum_update(self, model, momentum):
        for (src_name, src_parm), (tgt_name, tgt_parm) in zip(
            model.student.named_parameters(), model.teacher_onestage.named_parameters()
        ):
            tgt_parm.data.mul_(momentum).add_(src_parm.data, alpha=1 - momentum)

        # Upgrade running mean and running variance in the Normalizaiton layers, such as YOLO, where the BN
        # is trained from scratch
        if self.upgrade_norm:
            for (src_name, src_buffer), (tgt_name, tgt_buffer) in zip(
                    model.student.named_buffers(), model.teacher_onestage.named_buffers()
                ):
                if "running_mean" in src_name or "running_var" in src_name:
                    tgt_buffer.data.mul_(momentum).add_(src_buffer.data, alpha=1 - momentum)

@HOOKS.register_module()
class MeanTeacher_DualTeacehr_Buffer(MeanTeacher_DualTeacehr):
    """Load student weights from the YOLOX buffer, and update to senior
    """
    def __init__(
        self,
        momentum=0.999,
        interval=1,
        warm_up=100,
        decay_intervals=None,
        decay_factor=0.1,
        upgrade_norm=False
    ):
        super().__init__(momentum,
        interval,
        warm_up,
        decay_intervals,
        decay_factor,
        upgrade_norm)

    def before_train_iter(self, runner):
        """Update ema parameter every self.interval iterations, 
        after warm up epoches."""
        model = runner.model
        if is_module_wrapper(model):
            model = model.module

        if not runner.epoch > model.softlearning_after_epoch: 
            self.unupdated_iter = runner.iter
            return
        
        curr_step = runner.iter
        if curr_step % self.interval != 0:
            return
        
        # We warm up the momentum considering the instability at beginning
        momentum = min(
            self.momentum, 1 - (1 + self.warm_up) / (curr_step - self.unupdated_iter + 1 + self.warm_up)
        )
        runner.log_buffer.output["ema_momentum"] = momentum
        self.momentum_update(runner, model, momentum)

    def momentum_update(self, runner, model, momentum):

        with_ema_yolo = False
        for hook in runner.hooks:
            if hook.__class__.__name__ == "SubModulesExpMomentumEMAHook":
                model_buffers = hook.model_buffers
                param_ema_buffer = hook.param_ema_buffer
                with_ema_yolo = True

        # for name, parameter in self.model_parameters.items():
        #     # exclude num_tracking
        #     if parameter.dtype.is_floating_point:
        #         buffer_name = self.param_ema_buffer[name]
        #         buffer_parameter = self.model_buffers[buffer_name]
        #         buffer_parameter.mul_(1 - momentum).add_(
        #             parameter.data, alpha=momentum)

        if with_ema_yolo:
            for tgt_name, tgt_parm in model.teacher_onestage.named_parameters():
                buffer_name = param_ema_buffer["student."+tgt_name]
                buffer_parameter = model_buffers[buffer_name]
                tgt_parm.data.mul_(momentum).add_(buffer_parameter.data, alpha=1 - momentum)
            if self.upgrade_norm:
                for tgt_name, tgt_buffer in model.teacher_onestage.named_buffers():
                    if "running_mean" in tgt_name or "running_var" in tgt_name:
                        buffer_name = param_ema_buffer["student."+tgt_name]
                        buffer_parameter = model_buffers[buffer_name]
                        tgt_buffer.data.mul_(momentum).add_(buffer_parameter.data, alpha=1 - momentum)

        else:
            for (src_name, src_parm), (tgt_name, tgt_parm) in zip(
                model.student.named_parameters(), model.teacher_onestage.named_parameters()
            ):
                
                tgt_parm.data.mul_(momentum).add_(src_parm.data, alpha=1 - momentum)

            # Upgrade running mean and running variance in the Normalizaiton layers, such as YOLO, where the BN
            # is trained from scratch
            if self.upgrade_norm:
                for (src_name, src_buffer), (tgt_name, tgt_buffer) in zip(
                        model.student.named_buffers(), model.teacher_onestage.named_buffers()
                    ):
                    if "running_mean" in src_name or "running_var" in src_name:
                        tgt_buffer.data.mul_(momentum).add_(src_buffer.data, alpha=1 - momentum)
       

@HOOKS.register_module()
class MeanTeacher_MultiTeacher(MeanTeacher_DualTeacehr):
    """Support EMA on multiple senior model,
    iteratively update each senior model
    """
    def __init__(
        self,
        momentum=0.999,
        interval=1,
        warm_up=100,
        decay_intervals=None,
        decay_factor=0.1,
        upgrade_norm=False
    ):
        super().__init__(momentum,
            interval,
            warm_up,
            decay_intervals,
            decay_factor,
            upgrade_norm
            )
        self.updated_to = 1
    def before_run(self, runner):
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        for i in range(1,model.num_senior+1):
            assert hasattr(model, f"teacher_onestage_{i}")
        assert hasattr(model, "student")


    def before_train_iter(self, runner):
        """Update ema parameter every self.interval iterations, 
        after warm up epoches."""
        model = runner.model
        if is_module_wrapper(model):
            model = model.module

        if not runner.epoch > model.softlearning_after_epoch: 
            self.unupdated_iter = runner.iter
            return
        
        curr_step = runner.iter
        if curr_step % self.interval != 0:
            return
        
        # We warm up the momentum considering the instability at beginning
        momentum = min(
            self.momentum, 1 - (1 + self.warm_up) / (curr_step - self.unupdated_iter + 1 + self.warm_up)
        )
        runner.log_buffer.output["ema_momentum"] = momentum
        
        # Iteratively update
        model_teacher_onestage =  getattr(model, f"teacher_onestage_{self.updated_to}")
        self.momentum_update(model.student, model_teacher_onestage, 0)
        if self.updated_to == (model.num_senior):
            self.updated_to = 1
        else:
            self.updated_to += 1


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