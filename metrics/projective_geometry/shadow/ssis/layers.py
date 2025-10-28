import torch
from torch import nn
from torch.nn import Module, Parameter
from torch.nn import init

from detectron2.layers import Conv2d
from detectron2.layers import batched_nms
from detectron2.layers.batch_norm import get_norm

class _NewEmptyTensorOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, new_shape):
        ctx.shape = x.shape
        return x.new_empty(new_shape)

    @staticmethod
    def backward(ctx, grad):
        shape = ctx.shape
        return _NewEmptyTensorOp.apply(grad, shape), None


def ml_nms(boxlist, nms_thresh, max_proposals=-1,
           score_field="scores", label_field="labels"):
    """
    Performs non-maximum suppression on a boxlist, with scores specified
    in a boxlist field via score_field.

    Args:
        boxlist (detectron2.structures.Boxes):
        nms_thresh (float):
        max_proposals (int): if > 0, then only the top max_proposals are kept
            after non-maximum suppression
        score_field (str):
    """
    if nms_thresh <= 0:
        return boxlist
    boxes = boxlist.pred_boxes.tensor
    scores = boxlist.scores
    labels = boxlist.pred_classes
    keep = batched_nms(boxes, scores, labels, nms_thresh)
    if max_proposals > 0:
        keep = keep[: max_proposals]
    boxlist = boxlist[keep]
    return boxlist


class DFConv2d(nn.Module):
    """
    Deformable convolutional layer with configurable
    deformable groups, dilations and groups.

    Code is from:
    https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/layers/misc.py


    """
    def __init__(
            self,
            in_channels,
            out_channels,
            with_modulated_dcn=True,
            kernel_size=3,
            stride=1,
            groups=1,
            dilation=1,
            deformable_groups=1,
            bias=False,
            padding=None
    ):
        super(DFConv2d, self).__init__()
        if isinstance(kernel_size, (list, tuple)):
            assert isinstance(stride, (list, tuple))
            assert isinstance(dilation, (list, tuple))
            assert len(kernel_size) == 2
            assert len(stride) == 2
            assert len(dilation) == 2
            padding = (
                dilation[0] * (kernel_size[0] - 1) // 2,
                dilation[1] * (kernel_size[1] - 1) // 2
            )
            offset_base_channels = kernel_size[0] * kernel_size[1]
        else:
            padding = dilation * (kernel_size - 1) // 2
            offset_base_channels = kernel_size * kernel_size
        if with_modulated_dcn:
            from detectron2.layers.deform_conv import ModulatedDeformConv
            offset_channels = offset_base_channels * 3  # default: 27
            conv_block = ModulatedDeformConv
        else:
            from detectron2.layers.deform_conv import DeformConv
            offset_channels = offset_base_channels * 2  # default: 18
            conv_block = DeformConv
        self.offset = Conv2d(
            in_channels,
            deformable_groups * offset_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=1,
            dilation=dilation
        )
        for l in [self.offset, ]:
            nn.init.kaiming_uniform_(l.weight, a=1)
            torch.nn.init.constant_(l.bias, 0.)
        self.conv = conv_block(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            deformable_groups=deformable_groups,
            bias=bias
        )
        self.with_modulated_dcn = with_modulated_dcn
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.offset_split = offset_base_channels * deformable_groups * 2

    def forward(self, x, return_offset=False):
        if x.numel() > 0:
            if not self.with_modulated_dcn:
                offset_mask = self.offset(x)
                x = self.conv(x, offset_mask)
            else:
                offset_mask = self.offset(x)
                offset = offset_mask[:, :self.offset_split, :, :]
                mask = offset_mask[:, self.offset_split:, :, :].sigmoid()
                x = self.conv(x, offset, mask)
            if return_offset:
                return x, offset_mask
            return x
        # get output shape
        output_shape = [
            (i + 2 * p - (di * (k - 1) + 1)) // d + 1
            for i, p, di, k, d in zip(
                x.shape[-2:],
                self.padding,
                self.dilation,
                self.kernel_size,
                self.stride
            )
        ]
        output_shape = [x.shape[0], self.conv.weight.shape[0]] + output_shape
        return _NewEmptyTensorOp.apply(x, output_shape)


def conv_with_kaiming_uniform(
        norm=None, activation=None,
        use_deformable=False, use_sep=False):
    def make_conv(
        in_channels, out_channels, kernel_size, stride=1, dilation=1
    ):
        if use_deformable:
            conv_func = DFConv2d
        else:
            conv_func = Conv2d
        if use_sep:
            assert in_channels == out_channels
            groups = in_channels
        else:
            groups = 1
        conv = conv_func(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation,
            groups=groups,
            bias=(norm is None)
        )
        if not use_deformable:
            # Caffe2 implementation uses XavierFill, which in fact
            # corresponds to kaiming_uniform_ in PyTorch
            nn.init.kaiming_uniform_(conv.weight, a=1)
            if norm is None:
                nn.init.constant_(conv.bias, 0)
        module = [conv,]
        if norm is not None and len(norm) > 0:
            if norm == "GN":
                norm_module = nn.GroupNorm(32, out_channels)
            else:
                norm_module = get_norm(norm, out_channels)
            module.append(norm_module)
        if activation is not None:
            module.append(nn.ReLU(inplace=True))
        if len(module) > 1:
            return nn.Sequential(*module)
        return conv

    return make_conv


class NaiveGroupNorm(Module):
    r"""NaiveGroupNorm implements Group Normalization with the high-level matrix operations in PyTorch.
    It is a temporary solution to export GN by ONNX before the official GN can be exported by ONNX.
    The usage of NaiveGroupNorm is exactly the same as the official :class:`torch.nn.GroupNorm`.
    Args:
        num_groups (int): number of groups to separate the channels into
        num_channels (int): number of channels expected in input
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        affine: a boolean value that when set to ``True``, this module
            has learnable per-channel affine parameters initialized to ones (for weights)
            and zeros (for biases). Default: ``True``.

    Shape:
        - Input: :math:`(N, C, *)` where :math:`C=\text{num\_channels}`
        - Output: :math:`(N, C, *)` (same shape as input)

    Examples::

        >>> input = torch.randn(20, 6, 10, 10)
        >>> # Separate 6 channels into 3 groups
        >>> m = NaiveGroupNorm(3, 6)
        >>> # Separate 6 channels into 6 groups (equivalent with InstanceNorm)
        >>> m = NaiveGroupNorm(6, 6)
        >>> # Put all 6 channels into a single group (equivalent with LayerNorm)
        >>> m = NaiveGroupNorm(1, 6)
        >>> # Activating the module
        >>> output = m(input)

    .. _`Group Normalization`: https://arxiv.org/abs/1803.08494
    """
    __constants__ = ['num_groups', 'num_channels', 'eps', 'affine', 'weight',
                     'bias']

    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        super(NaiveGroupNorm, self).__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_channels))
            self.bias = Parameter(torch.Tensor(num_channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input):
        N, C, H, W = input.size()
        assert C % self.num_groups == 0
        input = input.reshape(N, self.num_groups, -1)
        mean = input.mean(dim=-1, keepdim=True)
        var = (input ** 2).mean(dim=-1, keepdim=True) - mean ** 2
        std = torch.sqrt(var + self.eps)

        input = (input - mean) / std
        input = input.reshape(N, C, H, W)
        if self.affine:
            input = input * self.weight.reshape(1, C, 1, 1) + self.bias.reshape(1, C, 1, 1)
        return input

    def extra_repr(self):
        return '{num_groups}, {num_channels}, eps={eps}, ' \
            'affine={affine}'.format(**self.__dict__)
