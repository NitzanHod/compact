import torch.nn as nn
from functions import small_linear


class SmallLinear(nn.Linear):
    def __init__(self, efficient: bool, in_features: int, out_features: int, bias: bool = True, device=None,
                 dtype=None, single_gpu=True):
        super().__init__(in_features=in_features, out_features=out_features, bias=bias, device=device, dtype=dtype)
        self.efficient = efficient
        self.single_gpu = single_gpu

    def forward(self, input):
        return small_linear(input, self.weight, self.bias, self.single_gpu)

    @classmethod
    def like(cls, linear, single_gpu: bool = True):
        # copy constructor - creates a CompAct Linear layer using the same weight and bias.
        small_lin = SmallLinear(efficient=True, in_features=linear.in_features, out_features=linear.out_features,
                                bias=linear.bias if isinstance(linear.bias, bool) else (linear.bias is not None),
                                device=linear.weight.device, dtype=linear.weight.dtype, single_gpu=single_gpu)
        small_lin.weight = linear.weight
        small_lin.bias = linear.bias
        return small_lin
