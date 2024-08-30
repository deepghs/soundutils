import numpy as np
import pytest
import torch

from soundutils.utils import np_conv1d
from test.testings import get_testfile


@pytest.fixture()
def t_waveform():
    return torch.load(get_testfile('assets', 'waveform_input.bin'), weights_only=True)


@pytest.fixture()
def waveform(t_waveform):
    return t_waveform.numpy()


@pytest.fixture()
def t_kernel():
    return torch.load(get_testfile('assets', 'kernel_input.bin'), weights_only=True)


@pytest.fixture()
def kernel(t_kernel):
    return t_kernel.numpy()


@pytest.mark.unittest
class TestUtilsConv:
    def test_data(self, kernel, t_kernel, waveform, t_waveform):
        assert isinstance(waveform, np.ndarray)
        assert waveform.shape == (1, 1, 514402)
        assert isinstance(kernel, np.ndarray)
        assert kernel.shape == (160, 1, 475)

        assert isinstance(t_waveform, torch.Tensor)
        assert t_waveform.shape == (1, 1, 514402)
        assert isinstance(t_kernel, torch.Tensor)
        assert t_kernel.shape == (160, 1, 475)

    def test_np_conv1d(self, kernel, t_kernel, waveform, t_waveform):
        expected = torch.nn.functional.conv1d(t_waveform, t_kernel, stride=441).numpy()
        actual = np_conv1d(waveform, kernel, stride=441)
        assert np.allclose(expected, actual, atol=1e-6)

    def test_np_conv1d_2dim(self, kernel, t_kernel, waveform, t_waveform):
        with pytest.raises(RuntimeError):
            _ = torch.nn.functional.conv1d(t_waveform[0], t_kernel[0], stride=441).numpy()
        with pytest.raises(RuntimeError):
            _ = np_conv1d(waveform[0], kernel[0], stride=441)
