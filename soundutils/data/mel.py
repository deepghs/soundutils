# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team and the librosa & torchaudio authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This module provides functions for converting between Hertz and Mel frequency scales.

The Mel scale is a perceptual scale of pitches judged by listeners to be equal in distance from one another.
This scale is often used in speech processing and audio analysis tasks.

The module supports three different Mel scale implementations:
1. HTK (Hidden Markov Model Toolkit)
2. Kaldi (Speech Recognition Toolkit)
3. Slaney (Malcolm Slaney's implementation)

Functions:
- hertz_to_mel: Convert frequency from Hertz to Mel scale
- mel_to_hertz: Convert frequency from Mel scale to Hertz

These functions support both single float values and numpy arrays as inputs.
"""

from typing import Union

import numpy as np


def hertz_to_mel(freq: Union[float, np.ndarray], mel_scale: str = "htk") -> Union[float, np.ndarray]:
    """
    Convert frequency from hertz to mels.

    This function transforms frequencies from the Hertz scale to the Mel scale.
    The Mel scale is a perceptual scale of pitches judged by listeners to be equal in distance from one another.

    :param freq: The frequency, or multiple frequencies, in hertz (Hz).
    :type freq: float or np.ndarray
    :param mel_scale: The mel frequency scale to use. Options are "htk", "kaldi", or "slaney".
    :type mel_scale: str, optional
    :return: The frequencies on the mel scale.
    :rtype: float or np.ndarray
    :raises ValueError: If mel_scale is not one of "htk", "slaney", or "kaldi".

    :Example:

    >>> hertz_to_mel(1000)
    1000.0
    >>> hertz_to_mel(np.array([440, 880]))
    array([548.68, 968.31])
    """

    if mel_scale not in ["slaney", "htk", "kaldi"]:
        raise ValueError('mel_scale should be one of "htk", "slaney" or "kaldi".')

    if mel_scale == "htk":
        return 2595.0 * np.log10(1.0 + (freq / 700.0))
    elif mel_scale == "kaldi":
        return 1127.0 * np.log(1.0 + (freq / 700.0))

    min_log_hertz = 1000.0
    min_log_mel = 15.0
    logstep = 27.0 / np.log(6.4)
    mels = 3.0 * freq / 200.0

    if isinstance(freq, np.ndarray):
        log_region = freq >= min_log_hertz
        mels[log_region] = min_log_mel + np.log(freq[log_region] / min_log_hertz) * logstep
    elif freq >= min_log_hertz:
        mels = min_log_mel + np.log(freq / min_log_hertz) * logstep

    return mels


def mel_to_hertz(mels: Union[float, np.ndarray], mel_scale: str = "htk") -> Union[float, np.ndarray]:
    """
    Convert frequency from mels to hertz.

    This function transforms frequencies from the Mel scale back to the Hertz scale.
    It is the inverse operation of hertz_to_mel.

    :param mels: The frequency, or multiple frequencies, in mels.
    :type mels: float or np.ndarray
    :param mel_scale: The mel frequency scale to use. Options are "htk", "kaldi", or "slaney".
    :type mel_scale: str, optional
    :return: The frequencies in hertz.
    :rtype: float or np.ndarray
    :raises ValueError: If mel_scale is not one of "htk", "slaney", or "kaldi".

    :Example:

    >>> mel_to_hertz(1000)
    1000.0
    >>> mel_to_hertz(np.array([548.68, 968.31]))
    array([440., 880.])
    """

    if mel_scale not in ["slaney", "htk", "kaldi"]:
        raise ValueError('mel_scale should be one of "htk", "slaney" or "kaldi".')

    if mel_scale == "htk":
        return 700.0 * (np.power(10, mels / 2595.0) - 1.0)
    elif mel_scale == "kaldi":
        return 700.0 * (np.exp(mels / 1127.0) - 1.0)

    min_log_hertz = 1000.0
    min_log_mel = 15.0
    logstep = np.log(6.4) / 27.0
    freq = 200.0 * mels / 3.0

    if isinstance(mels, np.ndarray):
        log_region = mels >= min_log_mel
        freq[log_region] = min_log_hertz * np.exp(logstep * (mels[log_region] - min_log_mel))
    elif mels >= min_log_mel:
        freq = min_log_hertz * np.exp(logstep * (mels - min_log_mel))

    return freq
