from typing import Optional

import numpy as np
from scipy import signal


def np_conv1d(input_: np.ndarray, weight: np.ndarray, bias: Optional[np.ndarray] = None,
              stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1):
    if input_.ndim < 3:
        raise RuntimeError('weight should have at least three dimensions')

    batch_size, in_channels, in_width = input_.shape
    out_channels, in_channels_per_group, kernel_width = weight.shape

    if in_channels % groups != 0:
        raise ValueError("in_channels must be divisible by groups")
    if out_channels % groups != 0:
        raise ValueError("out_channels must be divisible by groups")

    if isinstance(padding, str):
        if padding == 'valid':
            padding = 0
        elif padding == 'same':
            padding = ((kernel_width - 1) * dilation) // 2
        else:
            raise ValueError(f"Unsupported padding type - {padding!r}.")

    if padding > 0:
        pad_width = ((0, 0), (0, 0), (padding, padding))
        input_ = np.pad(input_, pad_width, mode='constant')

    out_width = (in_width + 2 * padding - dilation * (kernel_width - 1) - 1) // stride + 1
    output = np.zeros((batch_size, out_channels, out_width))
    for g in range(groups):
        for i in range(out_channels // groups):
            for b in range(batch_size):
                input_slice = input_[b, g * in_channels_per_group:(g + 1) * in_channels_per_group, :]
                weight_slice = weight[g * (out_channels // groups) + i]

                for c in range(in_channels_per_group):
                    output[b, g * (out_channels // groups) + i] += \
                        signal.correlate(input_slice[c], weight_slice[c], mode='valid', method='direct')[::stride]

    if bias is not None:
        output += bias.reshape((1, -1, 1))

    return output
