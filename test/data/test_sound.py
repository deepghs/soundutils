import re

import numpy as np
import pytest
import torch
import torchaudio
from hbutils.testing import isolated_directory, tmatrix

from soundutils.data import Sound
from soundutils.similarity import sound_spectral_centroid_distance, sound_mfcc_similarity
from ..testings import get_testfile


@pytest.mark.unittest
class TestDataSound:
    @pytest.mark.parametrize(['file', 'sample_rate', 'samples', 'time_'], [
        ('texas_short.wav', 44100, 31988, 0.7253514739229024),
        ('texas_assist.wav', 44100, 145474, 3.2987301587301587),
        ('texas_long.wav', 44100, 513927, 11.653673469387755),
    ])
    def test_1_channel(self, file, sample_rate, samples, time_):
        sound_file = get_testfile('assets', file)
        sound = Sound.open(sound_file)
        assert sound.sample_rate == sample_rate
        assert sound.samples == samples
        assert sound.time == pytest.approx(time_)
        assert sound.channels == 1

    @pytest.mark.parametrize(['file', 'sample_rate', 'samples', 'time_'], [
        ('stereo_sine_wave.wav', 16000, 32000, 2.0),
    ])
    def test_2_channels(self, file, sample_rate, samples, time_):
        sound_file = get_testfile('assets', file)
        sound = Sound.open(sound_file)
        assert sound.sample_rate == sample_rate
        assert sound.samples == samples
        assert sound.time == pytest.approx(time_)
        assert sound.channels == 2

    @pytest.mark.parametrize(['file'], [
        ('texas_short.wav',),
        ('texas_assist.wav',),
        ('texas_long.wav',),
        ('stereo_sine_wave.wav',),
        ('texas_long.mp3',),
        ('texas_long.flac',),
    ])
    def test_to_numpy(self, file):
        sound_file = get_testfile('assets', file)
        data, sample_rate = Sound.open(sound_file).to_numpy()
        t_data, t_sample_rate = torchaudio.load(sound_file)
        assert sample_rate == t_sample_rate
        assert np.isclose(data, t_data.numpy(), atol=1e-5).all()

    @pytest.mark.parametrize(['file'], [
        ('texas_short.wav',),
        ('texas_assist.wav',),
        ('texas_long.wav',),
        ('stereo_sine_wave.wav',),
        ('stereo_sine_wave_44100.wav',),
        ('stereo_sine_wave_3x_40_900.wav',),
        ('texas_long.mp3',),
        ('texas_long.flac',),
    ])
    def test_from_numpy(self, file):
        sound_file = get_testfile('assets', file)
        t_data, t_sample_rate = torchaudio.load(sound_file)
        sound = Sound.from_numpy(t_data.numpy(), t_sample_rate)
        assert sound_spectral_centroid_distance(sound, sound_file) < 1e-2

    def test_from_numpy_error(self):
        with pytest.raises(ValueError):
            Sound.from_numpy(np.random.randn(3, 4, 5), 16000)

    @pytest.mark.parametrize(['file'], [
        ('texas_short.wav',),
        ('texas_assist.wav',),
        ('texas_long.wav',),
        ('stereo_sine_wave.wav',),
        ('stereo_sine_wave_44100.wav',),
        ('stereo_sine_wave_3x_40_900.wav',),
        ('texas_long.mp3',),
        ('texas_long.flac',),
    ])
    def test_save(self, file):
        sound_file = get_testfile('assets', file)
        sound = Sound.open(sound_file)
        with isolated_directory():
            sound.save(file)
            assert sound_spectral_centroid_distance(sound_file, file) < 1

    def test_load_error(self):
        with pytest.raises(TypeError):
            Sound.load([])
        with pytest.raises(TypeError):
            Sound.load(None)
        with pytest.raises(TypeError):
            Sound.load(3.14)

    @pytest.mark.parametrize(['file', 'regex'], [
        ('texas_short.wav', r'<Sound 0x[a-f0-9]+, channels: 1, sample_rate: 44100, length: 0.725s \(31988 frames\)>'),
        ('texas_assist.wav', r'<Sound 0x[a-f0-9]+, channels: 1, sample_rate: 44100, length: 3.299s \(145474 frames\)>'),
        ('texas_long.wav', r'<Sound 0x[a-f0-9]+, channels: 1, sample_rate: 44100, length: 11.654s \(513927 frames\)>'),
        ('stereo_sine_wave.wav', r'<Sound 0x[a-f0-9]+, channels: 2, '
                                 r'sample_rate: 16000, length: 2.000s \(32000 frames\)>'),
    ])
    def test_repr(self, file, regex):
        sound_file = get_testfile('assets', file)
        sound = Sound.open(sound_file)
        assert re.fullmatch(regex, repr(sound)), \
            f'Repr {sound!r} not match with pattern {regex!r}.'

    @pytest.mark.parametrize(*tmatrix({
        'file': [
            'texas_short.wav',
            'texas_assist.wav',
            'texas_long.wav',
            'stereo_sine_wave.wav',
            'stereo_sine_wave_44100.wav',
            'stereo_sine_wave_3x_40_900.wav',
            'texas_long.flac',
        ],
        'sample_rate': [
            8000,
            16000,
            44100,
        ]
    }))
    def test_resample(self, file, sample_rate):
        sound_file = get_testfile('assets', file)
        sound = Sound.open(sound_file)
        resampler = torchaudio.transforms.Resample(sound.sample_rate, sample_rate)
        data, _ = sound.to_numpy()
        r_data = resampler(torch.from_numpy(data).type(torch.float32)).numpy()
        expected_new_sound = Sound.from_numpy(r_data, sample_rate)

        new_sound = sound.resample(sample_rate)
        assert new_sound.sample_rate == sample_rate
        assert sound_mfcc_similarity(
            new_sound, expected_new_sound,
            time_align='pad',
            resample_rate_align='min',
        ) >= 0.98
