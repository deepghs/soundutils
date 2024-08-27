from typing import Literal, Optional

import numpy as np
from scipy.signal import spectrogram

from .base import _align_sounds
from ..data import SoundTyping


def sound_spectral_centroid_distance(
        sound1: SoundTyping, sound2: SoundTyping,
        resample_rate_align: Literal['max', 'min', 'none'] = 'none',
        time_align: Literal['none', 'pad', 'prefix', 'resample_max', 'resample_min'] = 'none',
        channels_align: Literal['none'] = 'none',
        eps: Optional[float] = None
) -> float:
    (data1, sr1), (data2, sr2) = _align_sounds(
        sound1=sound1,
        sound2=sound2,
        resample_rate_align=resample_rate_align,
        time_align=time_align,
        channels_align=channels_align,
    )

    assert sr1 == sr2, 'Sample rate not match and not aligned, this must be a bug.'
    sr = sr1

    channels = data1.shape[0]
    distances = []
    eps = eps if eps is not None else np.finfo(data1.dtype).eps
    for ch in range(channels):
        _, _, Sxx1 = spectrogram(data1[ch], sr)
        _, _, Sxx2 = spectrogram(data2[ch], sr)

        Sxx1 += eps
        Sxx2 += eps

        centroid1 = np.sum(Sxx1 * np.arange(Sxx1.shape[0])[:, np.newaxis], axis=0) / np.sum(Sxx1, axis=0)
        centroid2 = np.sum(Sxx2 * np.arange(Sxx2.shape[0])[:, np.newaxis], axis=0) / np.sum(Sxx2, axis=0)

        distances.append(np.mean(np.abs(centroid1 - centroid2)))

    return np.mean(distances).item()
