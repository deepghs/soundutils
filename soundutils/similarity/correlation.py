from typing import Literal

import numpy as np

from soundutils.data import SoundTyping
from .base import _align_sounds


def sound_pearson_similarity(
        sound1: SoundTyping, sound2: SoundTyping,
        resample_rate_align: Literal['max', 'min', 'none'] = 'none',
        time_align: Literal['none', 'pad', 'prefix', 'resample_max', 'resample_min'] = 'none',
        channels_align: Literal['none'] = 'none',
) -> float:
    (data1, sr1), (data2, sr2) = _align_sounds(
        sound1=sound1,
        sound2=sound2,
        resample_rate_align=resample_rate_align,
        time_align=time_align,
        channels_align=channels_align,
    )

    # Average the correlation across channels
    correlation = np.mean(
        [np.corrcoef(data1[i, :], data2[i, :])[0, 1] for i in range(data1.shape[0])])
    return correlation.item()
