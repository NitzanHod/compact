from collections import defaultdict
from typing import DefaultDict, List, Optional

import torch
from torch.optim import Optimizer

def f(x):
    return x if x is None else x.shape

class SmallOptimizer(Optimizer):
    @torch._disable_dynamo
    def zero_grad(self, set_to_none: bool = True) -> None:
        foreach = self.defaults.get("foreach", False) or self.defaults.get(
            "fused", False
        )

        if not hasattr(self, "_zero_grad_profile_name"):
            self._patch_step_function()

        per_device_and_dtype_grads: Optional[
            DefaultDict[torch.device, DefaultDict[torch.dtype, List[torch.Tensor]]]
        ]
        if foreach:
            per_device_and_dtype_grads = defaultdict(lambda: defaultdict(list))
        else:
            per_device_and_dtype_grads = None

        with torch.autograd.profiler.record_function(self._zero_grad_profile_name):
            for group in self.param_groups:
                # parameters group has a small gradient
                for p in group["params"]:
                    if hasattr(p, "small_grad"):
                        assert (p.grad is None)
                        if p.small_grad is not None:
                            if set_to_none:
                                p.small_grad = None
                            else:
                                if p.small_grad.grad_fn is not None:
                                    p.small_grad.detach_()
                                else:
                                    p.small_grad.requires_grad_(False)
                                if not foreach or p.small_grad.is_sparse:
                                    p.small_grad.zero_()
                                else:
                                    assert per_device_and_dtype_grads is not None
                                    per_device_and_dtype_grads[p.small_grad.device][
                                        p.small_grad.dtype
                                    ].append(p.small_grad)

                    else:
                        # parameters in group are full rank
                        if p.grad is not None:
                            if set_to_none:
                                p.grad = None
                            else:
                                if p.grad.grad_fn is not None:
                                    p.grad.detach_()
                                else:
                                    p.grad.requires_grad_(False)
                                if not foreach or p.grad.is_sparse:
                                    p.grad.zero_()
                                else:
                                    assert per_device_and_dtype_grads is not None
                                    per_device_and_dtype_grads[p.grad.device][
                                        p.grad.dtype
                                    ].append(p.grad)

            if foreach:
                assert per_device_and_dtype_grads is not None
                for per_dtype_grads in per_device_and_dtype_grads.values():
                    for grads in per_dtype_grads.values():
                        torch._foreach_zero_(grads)
