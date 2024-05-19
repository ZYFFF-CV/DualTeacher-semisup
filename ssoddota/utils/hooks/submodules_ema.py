import math
from mmcv.runner.hooks import HOOKS, Hook
from mmdet.core.hook.ema import BaseEMAHook
from mmcv.parallel import is_module_wrapper

class SubModulesBaseEMAHook(Hook):
    """Exponential Moving Average Hook. Used in semi-supervised learning
    We do not hope the teacher models involed into the EMA update.
    Student model used only (or the specified module)

    Use Exponential Moving Average on all parameters of model in training
    process. All parameters have a ema backup, which update by the formula
    as below. EMAHook takes priority over EvalHook and CheckpointHook. Note,
    the original model parameters are actually saved in ema field after train.

    Args:
        momentum (float): The momentum used for updating ema parameter.
            Ema's parameter are updated with the formula:
           `ema_param = (1-momentum) * ema_param + momentum * cur_param`.
            Defaults to 0.0002.
        skip_buffers (bool): Whether to skip the model buffers, such as
            batchnorm running stats (running_mean, running_var), it does not
            perform the ema operation. Default to False.
        interval (int): Update ema parameter every interval iteration.
            Defaults to 1.
        resume_from (str, optional): The checkpoint path. Defaults to None.
        momentum_fun (func, optional): The function to change momentum
            during early iteration (also warmup) to help early training.
            It uses `momentum` as a constant. Defaults to None.
    """

    def __init__(self,
                 momentum=0.0002,
                 interval=1,
                 skip_buffers=False,
                 resume_from=None,
                 momentum_fun=None,
                 module_list=["student"],):
        assert 0 < momentum < 1
        self.momentum = momentum
        self.skip_buffers = skip_buffers
        self.interval = interval
        self.checkpoint = resume_from
        self.momentum_fun = momentum_fun
        self.module_list = module_list

    def before_run(self, runner):
        """To resume model with it's ema parameters more friendly.

        Register ema parameter as ``named_buffer`` to model.
        """
        model = runner.model
        if is_module_wrapper(model):
            model = model.module

        self.param_ema_buffer = {}
        self.model_buffers = {}
        for submodel_name in self.module_list:
            assert hasattr(model, submodel_name), f"Submodel {submodel_name} does not defiend"
            # submodel = getattr(model, submodel_name)

        if self.skip_buffers:
            self.model_parameters = dict(model.named_parameters())
        else:
            self.model_parameters = model.state_dict()
        for name, value in self.model_parameters.items():
            if sum([subname in name for subname in self.module_list]) == 0:
                continue
            # "." is not allowed in module's buffer name
            buffer_name = f"ema_{name.replace('.', '_')}"
            self.param_ema_buffer[name] = buffer_name
            model.register_buffer(buffer_name, value.data.clone())
        self.model_buffers = dict(model.named_buffers())
        if self.checkpoint is not None:
            runner.resume(self.checkpoint)

    def get_momentum(self, runner):
        return self.momentum_fun(runner.iter) if self.momentum_fun else \
                        self.momentum

    def after_train_iter(self, runner):
        """Update ema parameter every self.interval iterations."""
        if (runner.iter + 1) % self.interval != 0:
            return
        momentum = self.get_momentum(runner)
        for name, parameter in self.model_parameters.items():
            if sum([subname in name for subname in self.module_list]) == 0:
                continue
            # exclude num_tracking
            if parameter.dtype.is_floating_point:
                buffer_name = self.param_ema_buffer[name]
                buffer_parameter = self.model_buffers[buffer_name]
                buffer_parameter.mul_(1 - momentum).add_(
                    parameter.data, alpha=momentum)

    def after_train_epoch(self, runner):
        """We load parameter values from ema backup to model before the
        EvalHook."""
        self._swap_ema_parameters()

    def before_train_epoch(self, runner):
        """We recover model's parameter from ema backup after last epoch's
        EvalHook."""
        self._swap_ema_parameters()

    def _swap_ema_parameters(self):
        """Swap the parameter of model with parameter in ema_buffer."""
        for name, value in self.model_parameters.items():
            if sum([subname in name for subname in self.module_list]) == 0:
                continue
            temp = value.data.clone()
            ema_buffer = self.model_buffers[self.param_ema_buffer[name]]
            value.data.copy_(ema_buffer.data)
            ema_buffer.data.copy_(temp)

@HOOKS.register_module()
class SubModulesExpMomentumEMAHook(SubModulesBaseEMAHook):
    """EMAHook using exponential momentum strategy.
    Only ema upate for the specified submodule, mainly used in YOLOX

    Args:
        total_iter (int): The total number of iterations of EMA momentum.
           Defaults to 2000.
    """

    def __init__(self, module_list=["student"],total_iter=2000, **kwargs):
        super(SubModulesExpMomentumEMAHook, self).__init__(**kwargs)
        self.momentum_fun = lambda x: (1 - self.momentum) * math.exp(-(
            1 + x) / total_iter) + self.momentum
        self.module_list = module_list
        
    def after_train_iter(self, runner):
        """Update ema parameter every self.interval iterations."""
        if (runner.iter + 1) % self.interval != 0:
            return
        momentum = self.get_momentum(runner)
        for name, parameter in self.model_parameters.items():
            if sum([subname in name for subname in self.module_list]) == 0:
                continue
            # exclude num_tracking
            if parameter.dtype.is_floating_point:
                buffer_name = self.param_ema_buffer[name]
                buffer_parameter = self.model_buffers[buffer_name]
                buffer_parameter.mul_(1 - momentum).add_(
                    parameter.data, alpha=momentum)