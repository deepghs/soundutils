from typing import Literal

from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

from .base import _align_sounds
from ..data import SoundTyping


def sound_fastdtw(
        sound1: SoundTyping, sound2: SoundTyping,
        resample_rate_align: Literal['max', 'min', 'none'] = 'none',
        time_align: Literal['noncheck', 'pad', 'prefix', 'resample_max', 'resample_min'] = 'noncheck',
        channels_align: Literal['none'] = 'none',
        radius: int = 1,
) -> float:
    (data1, sr1), (data2, sr2) = _align_sounds(
        sound1=sound1,
        sound2=sound2,
        resample_rate_align=resample_rate_align,
        time_align=time_align,
        channels_align=channels_align,
    )
    return fastdtw(
        data1.T, data2.T,
        radius=radius,
        dist=euclidean,
    )
