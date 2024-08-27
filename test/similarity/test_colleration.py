import pytest

from soundutils.similarity import SoundLengthNotMatch, SoundChannelsNotMatch, SoundResampleRateNotMatch, \
    sound_pearson_similarity
from test.testings import get_testfile


@pytest.mark.unittest
class TestSimilarityCorrelation:
    @pytest.mark.parametrize(['file1', 'file2', 'v'], [
        ('texas_short.wav', 'texas_short.wav', 1.0),
        ('texas_short.wav', 'texas_long.wav', SoundLengthNotMatch),
        ('texas_long.wav', 'texas_long.wav', 1.0),
        ('texas_long.wav', 'texas_long.flac', 1.0),
        ('texas_long.wav', 'texas_long.mp3', 0.9982756956202339),
        ('stereo_sine_wave.wav', 'stereo_sine_wave_44100.wav', SoundResampleRateNotMatch),
        ('texas_short.wav', 'stereo_sine_wave_44100.wav', SoundChannelsNotMatch),
    ])
    def test_sound_pearson_similarity(self, file1, file2, v):
        file1 = get_testfile('assets', file1)
        file2 = get_testfile('assets', file2)
        if isinstance(v, type) and issubclass(v, BaseException):
            with pytest.raises(v):
                _ = sound_pearson_similarity(file1, file2)
        else:
            assert sound_pearson_similarity(file1, file2) == pytest.approx(v)

    @pytest.mark.parametrize(['file1', 'file2', 'v'], [
        ('texas_short.wav', 'texas_short.wav', 1.0),
        ('texas_short.wav', 'texas_long.wav', -0.023363902648302855),
        ('texas_long.wav', 'texas_short.wav', -0.023363902648302855),
        ('texas_long.wav', 'texas_long.wav', 1.0),
        ('texas_long.wav', 'texas_long.flac', 1.0),
        ('texas_long.wav', 'texas_long.mp3', 0.9982756956202339),
        ('stereo_sine_wave.wav', 'stereo_sine_wave_44100.wav', SoundResampleRateNotMatch),
        ('texas_short.wav', 'stereo_sine_wave_44100.wav', SoundChannelsNotMatch),
    ])
    def test_sound_pearson_similarity_pad(self, file1, file2, v):
        file1 = get_testfile('assets', file1)
        file2 = get_testfile('assets', file2)
        if isinstance(v, type) and issubclass(v, BaseException):
            with pytest.raises(v):
                _ = sound_pearson_similarity(file1, file2, time_align='pad')
        else:
            assert sound_pearson_similarity(file1, file2, time_align='pad') == pytest.approx(v)

    @pytest.mark.parametrize(['file1', 'file2', 'time_align', 'v'], [
        ('texas_short.wav', 'texas_short.wav', 'none', 1.0),
        ('texas_short.wav', 'texas_short.wav', 'pad', 1.0),
        ('texas_short.wav', 'texas_short.wav', 'prefix', 1.0),
        ('texas_short.wav', 'texas_short.wav', 'resample_max', 1.0),
        ('texas_short.wav', 'texas_short.wav', 'resample_min', 1.0),
        ('texas_short.wav', 'texas_short.wav', 'bullshit', ValueError),

        ('texas_short.wav', 'texas_long.wav', 'none', SoundLengthNotMatch),
        ('texas_short.wav', 'texas_long.wav', 'pad', -0.023363902648302855),
        ('texas_long.wav', 'texas_short.wav', 'pad', -0.023363902648302855),
        ('texas_short.wav', 'texas_long.wav', 'prefix', -0.05039022356294788),
        ('texas_short.wav', 'texas_long.wav', 'resample_max', -6.501929201342696e-05),
        ('texas_long.wav', 'texas_short.wav', 'resample_max', -6.501929201342696e-05),
        ('texas_short.wav', 'texas_long.wav', 'resample_min', -6.781880893649058e-05),
        ('texas_long.wav', 'texas_short.wav', 'resample_min', -6.781880893649058e-05),

        ('texas_long.wav', 'texas_long.wav', 'none', 1.0),
        ('texas_long.wav', 'texas_long.flac', 'none', 1.0),
        ('texas_long.wav', 'texas_long.mp3', 'none', 0.9982756956202339),
    ])
    def test_sound_pearson_similarity_time_align(self, file1, file2, time_align, v):
        file1 = get_testfile('assets', file1)
        file2 = get_testfile('assets', file2)
        if isinstance(v, type) and issubclass(v, BaseException):
            with pytest.raises(v):
                _ = sound_pearson_similarity(file1, file2, time_align=time_align)
        else:
            assert sound_pearson_similarity(file1, file2, time_align=time_align) == pytest.approx(v)
