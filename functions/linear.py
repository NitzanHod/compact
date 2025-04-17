import math

import torch


class SmallLinearFunction(torch.autograd.Function):
    """
    A CompAct Linear layer. Approximates nn.Linear via Gaussian Random Projection.
     The activation tensor is projected before being saved for backward.
    """
    @staticmethod
    def forward(ctx, input, weight, bias, single_gpu):
        output = input @ weight.T

        if bias is not None:
            output += bias  # += M

        # save tensors for backprop
        compressed_input = weight.projector.project(input)
        ctx.save_for_backward(compressed_input, weight, bias)
        ctx.single_gpu = single_gpu
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Approximated gradient is saved in small_grad instead of w.grad since torch does not allow w and w.grad
        to have different shapes.

        """
        # unpack saved tensors
        input, weight, bias = ctx.saved_tensors

        grad_input = grad_weight = grad_bias = None

        # calculate gradients
        if ctx.needs_input_grad[0]:
            grad_input = grad_output @ weight

        if ctx.needs_input_grad[1]:
            if len(grad_output.shape) == 3:
                small_grad = torch.einsum('bij,bik->jk', (grad_output, input))
            else:
                small_grad = torch.einsum('bi,bj->ij', (grad_output, input))

            if not ctx.single_gpu:
                # CompAct can be utilized to make DDP (data paralllelism) even better but that is currently a WIP.
                #  to support multi GPU we currently use the regular torch DDP, hence we have to store the gradients
                #  in w.grad whose shape must be as w, so we fill small_grad with zeros.
                m, n = weight.shape
                grad_weight = torch.cat(
                    [small_grad, torch.zeros(m, n - small_grad.shape[-1], device=small_grad.device,
                                              dtype=small_grad.dtype)], dim=1)

            elif (not hasattr(weight, "small_grad")) or (weight.small_grad is None):
                weight.small_grad = small_grad
            else:
                weight.small_grad += small_grad

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=0).sum(dim=0) if len(grad_output.shape) == 3 else grad_output.sum(dim=0)
        return grad_input, grad_weight, grad_bias, None


# wrapping the apply function, to allow keyword arguments
def small_linear(input, weight, bias, single_gpu):
    return SmallLinearFunction.apply(input, weight, bias, single_gpu)