import os
from datetime import datetime, timedelta
from typing import Tuple, Optional, Union

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import soundfile
from hbutils.string import plural_word
from matplotlib.ticker import FuncFormatter
from scipy import signal

SoundTyping = Union[str, os.PathLike, 'Sound']


class Sound:
    def __init__(self, data: np.ndarray, sample_rate: int):
        self._data = data
        self._sample_rate = sample_rate

    @property
    def samples(self) -> int:
        return self._data.shape[0]

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def time(self) -> float:
        return self.samples / self.sample_rate

    def _to_numpy(self) -> np.ndarray:
        if len(self._data.shape) == 1:
            data = self._data[..., None]
        else:
            data = self._data
        return data

    @property
    def channels(self) -> int:
        if len(self._data.shape) == 1:
            return 1
        else:
            return self._data.shape[-1]

    def resample(self, sample_rate) -> 'Sound':
        if sample_rate == self._sample_rate:
            return self

        resampled_length = int(self.samples * (sample_rate / self._sample_rate))
        resampled_data = signal.resample(self._data, resampled_length)

        return Sound(data=resampled_data, sample_rate=sample_rate)

    def crop(self, start_time: Optional[str] = None, end_time: Optional[str] = None) -> 'Sound':
        if start_time is None and end_time is None:
            return self

        start_sample = 0 if start_time is None else int(float(start_time) * self._sample_rate)
        end_sample = self.samples if end_time is None else int(float(end_time) * self._sample_rate)

        if start_sample < 0:
            start_sample = 0
        if end_sample > self.samples:
            end_sample = self.samples

        cropped_data = self._data[start_sample:end_sample]

        return Sound(data=cropped_data, sample_rate=self._sample_rate)

    def __repr__(self):
        return f'<{self.__class__.__name__} {hex(id(self))}, ' \
               f'channels: {self.channels!r}, sample_rate: {self._sample_rate}, ' \
               f'length: {self.time:.3f}s ({plural_word(self._data.shape[0], "frame")})>'

    def __getitem__(self, item):
        return self.__class__(
            data=self._to_numpy()[:, item],
            sample_rate=self._sample_rate,
        )

    def __iter__(self):
        sdata = self._to_numpy()
        for i in range(self.channels):
            yield self.__class__(
                data=sdata[:, i],
                sample_rate=self._sample_rate,
            )

    @classmethod
    def from_numpy(cls, data: np.ndarray, sample_rate: int) -> 'Sound':
        if len(data.shape) != 2:
            raise ValueError(f'Invalid sound data shape - {data.shape!r}.')

        data = data.T
        if data.shape[1] == 1:
            data = data[:, 0]
        return cls(data, sample_rate)

    def to_numpy(self) -> Tuple[np.ndarray, int]:
        # dump data to numpy format
        # it has been 100% aligned with torchaudio's loading result
        return self._to_numpy().T, self._sample_rate

    @classmethod
    def open(cls, sound_file: Union[str, os.PathLike]) -> 'Sound':
        data, sample_rate = soundfile.read(sound_file)
        return cls(data, sample_rate)

    def save(self, sound_file: Union[str, os.PathLike]):
        soundfile.write(sound_file, self._data, self._sample_rate)

    @classmethod
    def load(cls, sound: SoundTyping) -> 'Sound':
        if isinstance(sound, Sound):
            return sound
        elif isinstance(sound, (str, os.PathLike)):
            return cls.open(sound)
        else:
            raise TypeError(f'Unknown sound type - {sound!r}.')

    def plot(self, ax=None, title: Optional[str] = None):
        times = np.arange(self.samples) / float(self._sample_rate)
        base_time = datetime(1970, 1, 1)
        times = [base_time + timedelta(seconds=t) for t in times]
        times = mdates.date2num(times)

        ax = ax or plt.gca()
        data = self._to_numpy()
        for cid in range(self.channels):
            ax.plot(times, data[:, cid], label=f'Channel #{cid}', alpha=0.5)

        def _fmt_time(x, pos):
            dt, _ = mdates.num2date(x), pos
            return dt.strftime('%H:%M:%S') + f'.{int(dt.microsecond / 1000):03d}'

        ax.xaxis.set_major_formatter(FuncFormatter(_fmt_time))
        locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
        ax.xaxis.set_major_locator(locator)

        ax.set_xlabel('Time [hh:mm:ss.mmm]')
        ax.set_ylabel('Amplitude')
        ax.set_title(f'{title or "Audio Signal"}\n'
                     f'Channels: {self.channels}, Sample Rate: {self._sample_rate}\n'
                     f'Time: {self.time:.3f}s ({plural_word(self.samples, "frame")})\n')
        ax.legend()
