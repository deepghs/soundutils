import numpy as np
from scipy import signal


def conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    # 确保输入是3D
    if input.ndim != 3:
        raise ValueError("Input tensor must be 3-dimensional")

    # 获取输入和权重的形状
    batch_size, in_channels, in_width = input.shape
    out_channels, in_channels_per_group, kernel_width = weight.shape

    # 检查groups参数
    if in_channels % groups != 0:
        raise ValueError("in_channels must be divisible by groups")
    if out_channels % groups != 0:
        raise ValueError("out_channels must be divisible by groups")

    # 处理padding
    if isinstance(padding, str):
        if padding == 'valid':
            padding = 0
        elif padding == 'same':
            padding = ((kernel_width - 1) * dilation) // 2
        else:
            raise ValueError("Unsupported padding type")

    # 应用padding
    if padding > 0:
        pad_width = ((0, 0), (0, 0), (padding, padding))
        input = np.pad(input, pad_width, mode='constant')

    # 计算输出宽度
    out_width = (in_width + 2 * padding - dilation * (kernel_width - 1) - 1) // stride + 1

    # 初始化输出
    output = np.zeros((batch_size, out_channels, out_width))

    # 对每个group进行卷积
    for g in range(groups):
        for i in range(out_channels // groups):
            for b in range(batch_size):
                # 选择当前group的输入通道
                input_slice = input[b, g * in_channels_per_group:(g + 1) * in_channels_per_group, :]
                # 选择当前输出通道的权重
                weight_slice = weight[g * (out_channels // groups) + i]

                # 执行卷积
                for c in range(in_channels_per_group):
                    output[b, g * (out_channels // groups) + i] += \
                        signal.correlate(input_slice[c], weight_slice[c], mode='valid', method='direct')[::stride]

    # 应用bias
    if bias is not None:
        output += bias.reshape(1, -1, 1)

    return output


# 加载数据
import torch

waveform_input = torch.load('test/testfile/assets/waveform_input.bin', weights_only=True).numpy()
kernel_input = torch.load('test/testfile/assets/kernel_input.bin', weights_only=True).numpy()
orig_freq = 441

# 使用我们的conv1d函数
output = conv1d(waveform_input, kernel_input, stride=orig_freq)

# 比较结果
torch_output = torch.nn.functional.conv1d(
    torch.from_numpy(waveform_input),
    torch.from_numpy(kernel_input),
    stride=orig_freq
).numpy()

print("Max absolute difference:", np.max(np.abs(output - torch_output)))
print("Are the results equal?", np.allclose(output, torch_output, atol=1e-6))
