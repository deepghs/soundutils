from typing import Optional

import math
import numpy as np

from ..utils import np_conv1d


def _get_sinc_resample_kernel(
        orig_freq: int,
        new_freq: int,
        gcd: int,
        lowpass_filter_width: int = 6,
        rolloff: float = 0.99,
        resampling_method: str = "sinc_interp_hann",
        beta: Optional[float] = None,
):
    if not (int(orig_freq) == orig_freq and int(new_freq) == new_freq):
        raise Exception(
            "Frequencies must be of integer type to ensure quality resampling computation."
        )

    if resampling_method not in ["sinc_interp_hann", "sinc_interp_kaiser"]:
        raise ValueError("Invalid resampling method: {}".format(resampling_method))

    orig_freq = int(orig_freq) // gcd
    new_freq = int(new_freq) // gcd

    if lowpass_filter_width <= 0:
        raise ValueError("Low pass filter width should be positive.")
    base_freq = min(orig_freq, new_freq)
    base_freq *= rolloff

    width = math.ceil(lowpass_filter_width * orig_freq / base_freq)
    idx = np.arange(-width, width + orig_freq)[None, None] / orig_freq

    t = np.arange(0, -new_freq, -1)[:, None, None] / new_freq + idx
    t *= base_freq
    t = np.clip(t, -lowpass_filter_width, lowpass_filter_width)

    if resampling_method == "sinc_interp_hann":
        window = np.cos(t * math.pi / lowpass_filter_width / 2) ** 2
    else:
        if beta is None:
            beta = 14.769656459379492
        window = np.i0(beta * np.sqrt(1 - (t / lowpass_filter_width) ** 2)) / np.i0(beta)

    t *= math.pi
    scale = base_freq / orig_freq
    kernels = np.where(t == 0, 1.0, np.sin(t) / t)
    kernels *= window * scale
    return kernels.astype(np.float32), width


def _apply_sinc_resample_kernel(
        waveform: np.ndarray,
        orig_freq: int,
        new_freq: int,
        gcd: int,
        kernel: np.ndarray,
        width: int,
):
    if not np.issubdtype(waveform.dtype, np.floating):
        raise TypeError(f"Expected floating point type for waveform array, but received {waveform.dtype}.")

    orig_freq = int(orig_freq) // gcd
    new_freq = int(new_freq) // gcd

    shape = waveform.shape
    waveform = waveform.reshape(-1, shape[-1])
    num_wavs, length = waveform.shape
    waveform = np.pad(waveform, ((0, 0), (width, width + orig_freq)))

    resampled = np_conv1d(waveform[:, None], kernel, stride=orig_freq)
    resampled = resampled.transpose(0, 2, 1).reshape(num_wavs, -1)
    target_length = math.ceil(new_freq * length / orig_freq)
    resampled = resampled[..., :target_length]
    resampled = resampled.reshape(shape[:-1] + (resampled.shape[-1],))
    return resampled


def waveform_resample(
        waveform: np.ndarray,
        orig_freq: int,
        new_freq: int,
        lowpass_filter_width: int = 6,
        rolloff: float = 0.99,
        resampling_method: str = "sinc_interp_hann",
        beta: Optional[float] = None,
) -> np.ndarray:
    if orig_freq <= 0.0 or new_freq <= 0.0:
        raise ValueError("Original frequency and desired frequecy should be positive")

    if orig_freq == new_freq:
        return waveform

    gcd = math.gcd(int(orig_freq), int(new_freq))

    kernel, width = _get_sinc_resample_kernel(
        orig_freq,
        new_freq,
        gcd,
        lowpass_filter_width,
        rolloff,
        resampling_method,
        beta,
    )
    resampled = _apply_sinc_resample_kernel(waveform, orig_freq, new_freq, gcd, kernel, width)
    return resampled
