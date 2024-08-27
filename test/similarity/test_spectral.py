import pytest

from soundutils.similarity import SoundLengthNotMatch, SoundChannelsNotMatch, SoundResampleRateNotMatch, \
    sound_spectral_centroid_distance
from test.testings import get_testfile


@pytest.mark.unittest
class TestSimilaritySpectral:
    @pytest.mark.parametrize(['file1', 'file2', 'v'], [
        ('texas_short.wav', 'texas_short.wav', 0.0),
        ('texas_short.wav', 'texas_long.wav', SoundLengthNotMatch),
        ('texas_long.wav', 'texas_long.wav', 0.0),
        ('texas_long.wav', 'texas_long.flac', 0.0),
        ('texas_long.wav', 'texas_long.mp3', 1.0064432804796064),
        ('stereo_sine_wave.wav', 'stereo_sine_wave_44100.wav', SoundResampleRateNotMatch),
        ('texas_short.wav', 'stereo_sine_wave_44100.wav', SoundChannelsNotMatch),
    ])
    def test_sound_spectral_centroid_distance(self, file1, file2, v):
        file1 = get_testfile('assets', file1)
        file2 = get_testfile('assets', file2)
        if isinstance(v, type) and issubclass(v, BaseException):
            with pytest.raises(v):
                _ = sound_spectral_centroid_distance(file1, file2)
        else:
            assert sound_spectral_centroid_distance(file1, file2) == pytest.approx(v)

    @pytest.mark.parametrize(['file1', 'file2', 'v'], [
        ('texas_short.wav', 'texas_short.wav', 0.0),
        ('texas_short.wav', 'texas_long.wav', 39.91758021786038),
        ('texas_long.wav', 'texas_short.wav', 39.91758021786038),
        ('texas_long.wav', 'texas_long.wav', 0.0),
        ('texas_long.wav', 'texas_long.flac', 0.0),
        ('texas_long.wav', 'texas_long.mp3', 1.0064432804796064),
        ('stereo_sine_wave.wav', 'stereo_sine_wave_44100.wav', SoundResampleRateNotMatch),
        ('texas_short.wav', 'stereo_sine_wave_44100.wav', SoundChannelsNotMatch),
    ])
    def test_sound_spectral_centroid_distance_pad(self, file1, file2, v):
        file1 = get_testfile('assets', file1)
        file2 = get_testfile('assets', file2)
        if isinstance(v, type) and issubclass(v, BaseException):
            with pytest.raises(v):
                _ = sound_spectral_centroid_distance(file1, file2, time_align='pad')
        else:
            assert sound_spectral_centroid_distance(file1, file2, time_align='pad') == pytest.approx(v)

    @pytest.mark.parametrize(['file1', 'file2', 'time_align', 'v'], [
        ('texas_short.wav', 'texas_short.wav', 'none', 0.0),
        ('texas_short.wav', 'texas_short.wav', 'pad', 0.0),
        ('texas_short.wav', 'texas_short.wav', 'prefix', 0.0),
        ('texas_short.wav', 'texas_short.wav', 'resample_max', 0.0),
        ('texas_short.wav', 'texas_short.wav', 'resample_min', 0.0),
        ('texas_short.wav', 'texas_short.wav', 'bullshit', ValueError),

        ('texas_short.wav', 'texas_long.wav', 'none', SoundLengthNotMatch),
        ('texas_short.wav', 'texas_long.wav', 'pad', 39.91758021786038),
        ('texas_long.wav', 'texas_short.wav', 'pad', 39.91758021786038),
        ('texas_short.wav', 'texas_long.wav', 'prefix', 17.470158264068207),
        ('texas_short.wav', 'texas_long.wav', 'resample_max', 30.830995704432095),
        ('texas_long.wav', 'texas_short.wav', 'resample_max', 30.830995704432095),
        ('texas_short.wav', 'texas_long.wav', 'resample_min', 50.464673804842576),
        ('texas_long.wav', 'texas_short.wav', 'resample_min', 50.464673804842576),

        ('texas_long.wav', 'texas_long.wav', 'none', 0.0),
        ('texas_long.wav', 'texas_long.flac', 'none', 0.0),
        ('texas_long.wav', 'texas_long.mp3', 'none', 1.0064432804796064),
    ])
    def test_sound_spectral_centroid_distance_time_align(self, file1, file2, time_align, v):
        file1 = get_testfile('assets', file1)
        file2 = get_testfile('assets', file2)
        if isinstance(v, type) and issubclass(v, BaseException):
            with pytest.raises(v):
                _ = sound_spectral_centroid_distance(file1, file2, time_align=time_align)
        else:
            assert sound_spectral_centroid_distance(file1, file2, time_align=time_align) == pytest.approx(v)

    @pytest.mark.parametrize(['file1', 'file2', 'sr_align', 'v'], [
        ('stereo_sine_wave.wav', 'stereo_sine_wave.wav', 'none', 0.0),
        ('stereo_sine_wave.wav', 'stereo_sine_wave.wav', 'min', 0.0),
        ('stereo_sine_wave.wav', 'stereo_sine_wave.wav', 'max', 0.0),
        ('stereo_sine_wave.wav', 'stereo_sine_wave.wav', 'bullshit', ValueError),

        ('stereo_sine_wave.wav', 'stereo_sine_wave_44100.wav', 'none', SoundResampleRateNotMatch),
        ('stereo_sine_wave.wav', 'stereo_sine_wave_44100.wav', 'min', 0.0014548659720306609),
        ('stereo_sine_wave.wav', 'stereo_sine_wave_44100.wav', 'max', 0.0013925866190809756),

        ('stereo_sine_wave.wav', 'stereo_sine_wave_3x_40_900.wav', 'none', SoundResampleRateNotMatch),
        ('stereo_sine_wave.wav', 'stereo_sine_wave_3x_40_900.wav', 'min', 20.705757783547845),
        ('stereo_sine_wave.wav', 'stereo_sine_wave_3x_40_900.wav', 'max', 20.89222150977781),
    ])
    def test_sound_spectral_centroid_distance_sr_align(self, file1, file2, sr_align, v):
        file1 = get_testfile('assets', file1)
        file2 = get_testfile('assets', file2)
        if isinstance(v, type) and issubclass(v, BaseException):
            with pytest.raises(v):
                _ = sound_spectral_centroid_distance(file1, file2, time_align='pad', resample_rate_align=sr_align)
        else:
            assert sound_spectral_centroid_distance(file1, file2, time_align='pad', resample_rate_align=sr_align) \
                   == pytest.approx(v)
