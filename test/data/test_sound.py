import numpy as np
import pytest
import torchaudio

from soundutils.data import Sound
from ..testings import get_testfile


@pytest.mark.unittest
class TestDataSound:
    @pytest.mark.parametrize(['file', 'sample_rate', 'samples', 'time_'], [
        ('texas_short.wav', 44100, 31988, 0.7253514739229024),
        ('texas_assist.wav', 44100, 145474, 3.2987301587301587),
        ('texas_long.wav', 44100, 513927, 11.653673469387755),
    ])
    def test_one_channel(self, file, sample_rate, samples, time_):
        sound_file = get_testfile('assets', file)
        sound = Sound.open(sound_file)
        assert sound.sample_rate == sample_rate
        assert sound.samples == samples
        assert sound.time == pytest.approx(time_)
        assert sound.channels == 1

    @pytest.mark.parametrize(['file'], [
        ('texas_short.wav',),
        ('texas_assist.wav',),
        ('texas_long.wav',),
        ('stereo_sine_wave.wav',),
        ('texas_long.mp3',),
        ('texas_long.flac',),
    ])
    def test_one_channel_to_numpy(self, file):
        sound_file = get_testfile('assets', file)
        data, sample_rate = Sound.open(sound_file).to_numpy()
        t_data, t_sample_rate = torchaudio.load(sound_file)
        assert sample_rate == t_sample_rate
        assert np.isclose(data, t_data.numpy(), atol=1e-5).all()
