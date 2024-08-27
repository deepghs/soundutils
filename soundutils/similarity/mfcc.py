from typing import Literal

import librosa
import numpy as np
from scipy.spatial.distance import cosine

from .base import _align_sounds
from ..data import SoundTyping


def sound_mfcc_similarity(
        sound1: SoundTyping, sound2: SoundTyping,
        n_mfcc: int = 13, mode: Literal['flat', 'mean'] = 'flat',
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

    similarities = []
    for ch in range(data1.shape[0]):
        mfcc1 = librosa.feature.mfcc(y=data1[ch], sr=sr1, n_mfcc=n_mfcc)
        mfcc2 = librosa.feature.mfcc(y=data2[ch], sr=sr2, n_mfcc=n_mfcc)
        if mode == 'flat':
            mfcc1_feat = mfcc1.flatten()
            mfcc2_feat = mfcc2.flatten()
        elif mode == 'mean':
            mfcc1_feat = np.mean(mfcc1, axis=-1)
            mfcc2_feat = np.mean(mfcc2, axis=-1)
        else:
            raise ValueError(f'Invalid mode for MFCC - {mode!r}.')
        similarities.append(1 - cosine(mfcc1_feat, mfcc2_feat))

    return np.mean(similarities).item()
