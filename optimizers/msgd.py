# copy dependencies from transformers/optimization.py
import math
import wandb
import warnings
from typing import Callable, Iterable, Tuple

import torch
from torch import nn

from transformers.utils.versions import require_version

from .small_optimizer import SmallOptimizer


class MSGD(SmallOptimizer):
    """
    Implements Adam algorithm with weight decay fix as introduced in [Decoupled Weight Decay
    Regularization](https://arxiv.org/abs/1711.05101).

    Parameters:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*, defaults to 0.001):
            The learning rate to use.
        betas (`Tuple[float,float]`, *optional*, defaults to `(0.9, 0.999)`):
            Adam's betas parameters (b1, b2).
        eps (`float`, *optional*, defaults to 1e-06):
            Adam's epsilon for numerical stability.
        weight_decay (`float`, *optional*, defaults to 0.0):
            Decoupled weight decay to apply.
        correct_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to correct bias in Adam (for instance, in Bert TF repository they use `False`).
        no_deprecation_warning (`bool`, *optional*, defaults to `False`):
            A flag used to disable the deprecation warning (set to `True` to disable the warning).
    """

    def __init__(
            self,
            params: Iterable[nn.parameter.Parameter],
            lr: float = 1e-3,
            momentum: float = 0.0,
            dampening: float = 0.0,
            weight_decay: float = 0.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = {"lr": lr, "momentum": momentum, "weight_decay": weight_decay, "dampening": dampening}
        super().__init__(params, defaults)


    @torch.no_grad()
    def step(self, closure: Callable = None, iter = 0, global_rank=None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                has_small_grad = hasattr(p, "small_grad")
                if (p.grad is None) and (not has_small_grad):
                    continue

                grad = p.grad if not has_small_grad else p.small_grad
                # if grad is padded with zeros - remove them
                if hasattr(p, 'projector'):
                    grad = p.projector.proccess_grad(grad)


                if grad.is_sparse:
                    raise RuntimeError("MSGD does not support sparse gradients, please consider SparseAdam instead")

                if group["weight_decay"] > 0.0:
                    grad.add_(p, alpha= group["weight_decay"])

                state = self.state[p]

                if "step" not in state:
                    state["step"] = 0

                # State initialization
                if "exp_avg" not in state:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(grad)

                exp_avg = state["exp_avg"]
                momentum, dampening = group["momentum"], group["dampening"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(momentum).add_(grad, alpha=(1.0 - dampening))

                step_size = group["lr"]

                # compute norm gradient
                norm_grad = exp_avg

                # Project Gradient Back
                if hasattr(p, "projector"):
                    norm_grad = p.projector.project_back(norm_grad)

                if hasattr(p, "log_grad") and global_rank==0 and (iter%100 == 0):
                    wandb.log({'mean_'+p.log_name: norm_grad.mean(), 'var_'+p.log_name: norm_grad.var(),
                               'min_'+p.log_name: norm_grad.min(), 'max_'+p.log_name: norm_grad.max(),
                               'norm_'+p.log_name: norm_grad.norm()}, step=iter)

                p.add_(norm_grad, alpha=-step_size)

        return loss
