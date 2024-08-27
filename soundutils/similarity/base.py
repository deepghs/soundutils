from typing import Tuple, Literal

import numpy as np
from hbutils.string import plural_word
from scipy.signal import resample

from ..data import Sound, SoundTyping


class SoundAlignError(Exception):
    pass


class SoundChannelsNotMatch(SoundAlignError):
    pass


class SoundResampleRateNotMatch(SoundAlignError):
    pass


class SoundLengthNotMatch(SoundAlignError):
    pass


def _align_sounds(
        sound1: SoundTyping, sound2: SoundTyping,
        resample_rate_align: Literal['max', 'min', 'none'] = 'none',
        time_align: Literal['pad', 'resample', 'none'] = 'none',
        channels_align: Literal['none'] = 'none',
) -> Tuple[np.ndarray, np.ndarray]:
    sound1, sound2 = Sound.load(sound1), Sound.load(sound2)
    if channels_align == 'none':
        if sound1.channels != sound2.channels:
            raise SoundChannelsNotMatch(f'Sound channels not match - {sound1.channels!r} vs {sound2.channels!r}.')
    else:
        raise ValueError(f'Invalid channels align mode - {channels_align!r}.')

    data1, sr1 = sound1.to_numpy()
    data2, sr2 = sound2.to_numpy()

    if resample_rate_align == 'none':
        if sr1 != sr2:
            raise SoundResampleRateNotMatch(f'Sound resample rate not match - {sr1!r} vs {sr2!r}.')
    else:
        # Determine a common sample rate, here using the higher of the two
        if resample_rate_align == 'max':
            c_sr = max(sr1, sr2)
        elif resample_rate_align == 'min':
            c_sr = min(sr1, sr2)
        else:
            raise ValueError(f'Invalid resample rate align mode - {resample_rate_align!r}.')

        if sr1 != c_sr:
            num_samples = int(sound1.time * c_sr)
            data1 = resample(data1, num_samples)
            sr1 = c_sr
        if sr2 != c_sr:
            num_samples = int(sound2.time * c_sr)
            data2 = resample(data2, num_samples)
            sr2 = c_sr

    if time_align == 'none':
        if data1.shape[-1] != data2.shape[-1]:
            raise SoundLengthNotMatch('Sound length not match - '
                                      f'{data1.shape[-1] / sr1:.3f}s ({plural_word(data1.shape[-1], "frame")}) vs '
                                      f'{data2.shape[-1] / sr2:.3f}s ({plural_word(data2.shape[-1], "frame")}).')
    else:
        # shape of data1 and data2: (channels, frames)
        # TODO: support 3 modes of time_align:
        # * 'pad', pad the sound data with fewer frames with all 0 constants
        # * 'resample_max', resample the shorter sound data to the longer one's frames
        # * 'resample_min', resample the longer sound data to the shorter one's frames
        raise NotImplementedError

    return data1, data2
