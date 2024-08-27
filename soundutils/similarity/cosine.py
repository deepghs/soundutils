from typing import Literal

import numpy as np
from scipy.stats import cosine

from .base import _align_sounds
from ..data import SoundTyping


def sound_cosine_similarity(
        sound1: SoundTyping, sound2: SoundTyping,
        resample_rate_align: Literal['max', 'min', 'none'] = 'none',
        time_align: Literal['pad', 'resample', 'none'] = 'none',
        channels_align: Literal['none'] = 'none',
) -> float:
    data1, data2 = _align_sounds(
        sound1=sound1,
        sound2=sound2,
        resample_rate_align=resample_rate_align,
        time_align=time_align,
        channels_align=channels_align,
    )

    # Cosine similarity
    cosine_similarity = np.mean(
        [cosine(data2[:, i], data1[:, i]) for i in range(data1.shape[0])])
    return cosine_similarity.item()
