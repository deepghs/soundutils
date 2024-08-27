from typing import Literal

import numpy as np

from .base import _align_sounds
from ..data import SoundTyping


def sound_mse(
        sound1: SoundTyping, sound2: SoundTyping,
        resample_rate_align: Literal['max', 'min', 'none'] = 'none',
        time_align: Literal['none', 'pad', 'prefix', 'resample_max', 'resample_min'] = 'none',
        channels_align: Literal['none'] = 'none',
        p: float = 2.0
) -> float:
    (data1, sr1), (data2, sr2) = _align_sounds(
        sound1=sound1,
        sound2=sound2,
        resample_rate_align=resample_rate_align,
        time_align=time_align,
        channels_align=channels_align,
    )

    return np.mean((data1 - data2) ** p).item()


def sound_rmse(
        sound1: SoundTyping, sound2: SoundTyping,
        resample_rate_align: Literal['max', 'min', 'none'] = 'none',
        time_align: Literal['none', 'pad', 'prefix', 'resample_max', 'resample_min'] = 'none',
        channels_align: Literal['none'] = 'none',
        p: float = 2.0,
) -> float:
    (data1, sr1), (data2, sr2) = _align_sounds(
        sound1=sound1,
        sound2=sound2,
        resample_rate_align=resample_rate_align,
        time_align=time_align,
        channels_align=channels_align,
    )
    return (np.mean((data1 - data2) ** p) ** (1.0 / p)).item()
