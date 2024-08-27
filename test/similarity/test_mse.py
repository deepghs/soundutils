import pytest

from soundutils.similarity import SoundLengthNotMatch, SoundChannelsNotMatch, SoundResampleRateNotMatch, \
    sound_mse, sound_rmse
from test.testings import get_testfile


@pytest.mark.unittest
class TestSimilarityMSE:
    @pytest.mark.parametrize(['file1', 'file2', 'v'], [
        ('texas_short.wav', 'texas_short.wav', 0.0),
        ('texas_short.wav', 'texas_long.wav', SoundLengthNotMatch),
        ('texas_long.wav', 'texas_long.wav', 0.0),
        ('texas_long.wav', 'texas_long.flac', 0.0),
        ('texas_long.wav', 'texas_long.mp3', 3.3204135745854366e-05),
        ('stereo_sine_wave.wav', 'stereo_sine_wave_44100.wav', SoundResampleRateNotMatch),
        ('texas_short.wav', 'stereo_sine_wave_44100.wav', SoundChannelsNotMatch),
    ])
    def test_sound_mse(self, file1, file2, v):
        file1 = get_testfile('assets', file1)
        file2 = get_testfile('assets', file2)
        if isinstance(v, type) and issubclass(v, BaseException):
            with pytest.raises(v):
                _ = sound_mse(file1, file2)
        else:
            assert sound_mse(file1, file2) == pytest.approx(v)

    @pytest.mark.parametrize(['file1', 'file2', 'v'], [
        ('texas_short.wav', 'texas_short.wav', 0.0),
        ('texas_short.wav', 'texas_long.wav', 0.010523605958231633),
        ('texas_long.wav', 'texas_short.wav', 0.010523605958231633),
        ('texas_long.wav', 'texas_long.wav', 0.0),
        ('texas_long.wav', 'texas_long.flac', 0.0),
        ('texas_long.wav', 'texas_long.mp3', 3.3204135745854366e-05),
        ('stereo_sine_wave.wav', 'stereo_sine_wave_44100.wav', SoundResampleRateNotMatch),
        ('texas_short.wav', 'stereo_sine_wave_44100.wav', SoundChannelsNotMatch),
    ])
    def test_sound_mse_pad(self, file1, file2, v):
        file1 = get_testfile('assets', file1)
        file2 = get_testfile('assets', file2)
        if isinstance(v, type) and issubclass(v, BaseException):
            with pytest.raises(v):
                _ = sound_mse(file1, file2, time_align='pad')
        else:
            assert sound_mse(file1, file2, time_align='pad') == pytest.approx(v)

    @pytest.mark.parametrize(['file1', 'file2', 'time_align', 'v'], [
        ('texas_short.wav', 'texas_short.wav', 'none', 0.0),
        ('texas_short.wav', 'texas_short.wav', 'pad', 0.0),
        ('texas_short.wav', 'texas_short.wav', 'prefix', 0.0),
        ('texas_short.wav', 'texas_short.wav', 'resample_max', 0.0),
        ('texas_short.wav', 'texas_short.wav', 'resample_min', 0.0),
        ('texas_short.wav', 'texas_short.wav', 'bullshit', ValueError),

        ('texas_short.wav', 'texas_long.wav', 'none', SoundLengthNotMatch),
        ('texas_short.wav', 'texas_long.wav', 'pad', 0.010523605958231633),
        ('texas_long.wav', 'texas_short.wav', 'pad', 0.010523605958231633),
        ('texas_short.wav', 'texas_long.wav', 'prefix', 0.04764499593602056),
        ('texas_short.wav', 'texas_long.wav', 'resample_max', 0.0219781205067706),
        ('texas_long.wav', 'texas_short.wav', 'resample_max', 0.0219781205067706),
        ('texas_short.wav', 'texas_long.wav', 'resample_min', 0.021199713154406444),
        ('texas_long.wav', 'texas_short.wav', 'resample_min', 0.021199713154406444),

        ('texas_long.wav', 'texas_long.wav', 'none', 0.0),
        ('texas_long.wav', 'texas_long.flac', 'none', 0.0),
        ('texas_long.wav', 'texas_long.mp3', 'none', 3.3204135745854366e-05),
    ])
    def test_sound_mse_time_align(self, file1, file2, time_align, v):
        file1 = get_testfile('assets', file1)
        file2 = get_testfile('assets', file2)
        if isinstance(v, type) and issubclass(v, BaseException):
            with pytest.raises(v):
                _ = sound_mse(file1, file2, time_align=time_align)
        else:
            assert sound_mse(file1, file2, time_align=time_align) == pytest.approx(v)

    @pytest.mark.parametrize(['file1', 'file2', 'v'], [
        ('texas_short.wav', 'texas_short.wav', 0.0),
        ('texas_short.wav', 'texas_long.wav', SoundLengthNotMatch),
        ('texas_long.wav', 'texas_long.wav', 0.0),
        ('texas_long.wav', 'texas_long.flac', 0.0),
        ('texas_long.wav', 'texas_long.mp3', 0.005762302989764974),
        ('stereo_sine_wave.wav', 'stereo_sine_wave_44100.wav', SoundResampleRateNotMatch),
        ('texas_short.wav', 'stereo_sine_wave_44100.wav', SoundChannelsNotMatch),
    ])
    def test_sound_rmse(self, file1, file2, v):
        file1 = get_testfile('assets', file1)
        file2 = get_testfile('assets', file2)
        if isinstance(v, type) and issubclass(v, BaseException):
            with pytest.raises(v):
                _ = sound_rmse(file1, file2)
        else:
            assert sound_rmse(file1, file2) == pytest.approx(v)

    @pytest.mark.parametrize(['file1', 'file2', 'v'], [
        ('texas_short.wav', 'texas_short.wav', 0.0),
        ('texas_short.wav', 'texas_long.wav', 0.10258462827456964),
        ('texas_long.wav', 'texas_short.wav', 0.10258462827456964),
        ('texas_long.wav', 'texas_long.wav', 0.0),
        ('texas_long.wav', 'texas_long.flac', 0.0),
        ('texas_long.wav', 'texas_long.mp3', 0.005762302989764974),
        ('stereo_sine_wave.wav', 'stereo_sine_wave_44100.wav', SoundResampleRateNotMatch),
        ('texas_short.wav', 'stereo_sine_wave_44100.wav', SoundChannelsNotMatch),
    ])
    def test_sound_rmse_pad(self, file1, file2, v):
        file1 = get_testfile('assets', file1)
        file2 = get_testfile('assets', file2)
        if isinstance(v, type) and issubclass(v, BaseException):
            with pytest.raises(v):
                _ = sound_rmse(file1, file2, time_align='pad')
        else:
            assert sound_rmse(file1, file2, time_align='pad') == pytest.approx(v)

    @pytest.mark.parametrize(['file1', 'file2', 'time_align', 'v'], [
        ('texas_short.wav', 'texas_short.wav', 'none', 0.0),
        ('texas_short.wav', 'texas_short.wav', 'pad', 0.0),
        ('texas_short.wav', 'texas_short.wav', 'prefix', 0.0),
        ('texas_short.wav', 'texas_short.wav', 'resample_max', 0.0),
        ('texas_short.wav', 'texas_short.wav', 'resample_min', 0.0),
        ('texas_short.wav', 'texas_short.wav', 'bullshit', ValueError),

        ('texas_short.wav', 'texas_long.wav', 'none', SoundLengthNotMatch),
        ('texas_short.wav', 'texas_long.wav', 'pad', 0.10258462827456964),
        ('texas_long.wav', 'texas_short.wav', 'pad', 0.10258462827456964),
        ('texas_short.wav', 'texas_long.wav', 'prefix', 0.21827733720205714),
        ('texas_short.wav', 'texas_long.wav', 'resample_max', 0.14825019563822034),
        ('texas_long.wav', 'texas_short.wav', 'resample_max', 0.14825019563822034),
        ('texas_short.wav', 'texas_long.wav', 'resample_min', 0.14560121275046595),
        ('texas_long.wav', 'texas_short.wav', 'resample_min', 0.14560121275046595),

        ('texas_long.wav', 'texas_long.wav', 'none', 0.0),
        ('texas_long.wav', 'texas_long.flac', 'none', 0.0),
        ('texas_long.wav', 'texas_long.mp3', 'none', 0.005762302989764974),
    ])
    def test_sound_rmse_time_align(self, file1, file2, time_align, v):
        file1 = get_testfile('assets', file1)
        file2 = get_testfile('assets', file2)
        if isinstance(v, type) and issubclass(v, BaseException):
            with pytest.raises(v):
                _ = sound_rmse(file1, file2, time_align=time_align)
        else:
            assert sound_rmse(file1, file2, time_align=time_align) == pytest.approx(v)
