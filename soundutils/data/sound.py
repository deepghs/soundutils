from typing import Tuple

import numpy as np
import soundfile


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

    def getchannels(self, channels):
        return self._to_numpy()[:, channels]

    def to_numpy(self) -> Tuple[np.ndarray, int]:
        return self._to_numpy().T, self._sample_rate

    @classmethod
    def from_numpy(cls, data: np.ndarray, sample_rate: int) -> 'Sound':
        if len(data.shape) != 2:
            raise ValueError(f'Invalid sound data shape - {data.shape!r}.')

        data = data.T
        if data.shape[0] == 1:
            data = data[:, 0]
        return cls(data, sample_rate)

    @classmethod
    def open(cls, sound_file: str) -> 'Sound':
        data, sample_rate = soundfile.read(sound_file)
        return cls(data, sample_rate)

    def save(self, sound_file: str):
        soundfile.write(sound_file, self._data, self._sample_rate)

    def __repr__(self):
        return f'<{self.__class__.__name__} {hex(id(self))}, ' \
               f'samples: {self._data.shape[0]}, sample_rate: {self._sample_rate}, ' \
               f'time: {self.time:.3f}s>'

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
