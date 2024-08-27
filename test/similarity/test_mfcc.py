import pytest

from soundutils.similarity import SoundLengthNotMatch, SoundChannelsNotMatch, SoundResampleRateNotMatch, \
    sound_mfcc_similarity
from test.testings import get_testfile


@pytest.mark.unittest
class TestSimilarityMFCC:
    @pytest.mark.parametrize(['file1', 'file2', 'v'], [
        ('texas_short.wav', 'texas_short.wav', 1.0),
        ('texas_short.wav', 'texas_long.wav', SoundLengthNotMatch),
        ('texas_long.wav', 'texas_long.wav', 1.0),
        ('texas_long.wav', 'texas_long.flac', 1.0),
        ('texas_long.wav', 'texas_long.mp3', 0.9987497286333322),
        ('stereo_sine_wave.wav', 'stereo_sine_wave_44100.wav', SoundResampleRateNotMatch),
        ('texas_short.wav', 'stereo_sine_wave_44100.wav', SoundChannelsNotMatch),
    ])
    def test_sound_mfcc_similarity(self, file1, file2, v):
        file1 = get_testfile('assets', file1)
        file2 = get_testfile('assets', file2)
        if isinstance(v, type) and issubclass(v, BaseException):
            with pytest.raises(v):
                _ = sound_mfcc_similarity(file1, file2)
        else:
            assert sound_mfcc_similarity(file1, file2) == pytest.approx(v)

    @pytest.mark.parametrize(['file1', 'file2', 'v'], [
        ('texas_short.wav', 'texas_short.wav', 1.0),
        ('texas_short.wav', 'texas_long.wav', 0.8899687300652068),
        ('texas_long.wav', 'texas_short.wav', 0.8899687300652068),
        ('texas_long.wav', 'texas_long.wav', 1.0),
        ('texas_long.wav', 'texas_long.flac', 1.0),
        ('texas_long.wav', 'texas_long.mp3', 0.9987497286333322),
        ('stereo_sine_wave.wav', 'stereo_sine_wave_44100.wav', SoundResampleRateNotMatch),
        ('texas_short.wav', 'stereo_sine_wave_44100.wav', SoundChannelsNotMatch),
    ])
    def test_sound_mfcc_similarity_pad(self, file1, file2, v):
        file1 = get_testfile('assets', file1)
        file2 = get_testfile('assets', file2)
        if isinstance(v, type) and issubclass(v, BaseException):
            with pytest.raises(v):
                _ = sound_mfcc_similarity(file1, file2, time_align='pad')
        else:
            assert sound_mfcc_similarity(file1, file2, time_align='pad') == pytest.approx(v)

    @pytest.mark.parametrize(['file1', 'file2', 'time_align', 'v'], [
        ('texas_short.wav', 'texas_short.wav', 'none', 1.0),
        ('texas_short.wav', 'texas_short.wav', 'pad', 1.0),
        ('texas_short.wav', 'texas_short.wav', 'prefix', 1.0),
        ('texas_short.wav', 'texas_short.wav', 'resample_max', 1.0),
        ('texas_short.wav', 'texas_short.wav', 'resample_min', 1.0),
        ('texas_short.wav', 'texas_short.wav', 'bullshit', ValueError),

        ('texas_short.wav', 'texas_long.wav', 'none', SoundLengthNotMatch),
        ('texas_short.wav', 'texas_long.wav', 'pad', 0.8899687300652068),
        ('texas_long.wav', 'texas_short.wav', 'pad', 0.8899687300652068),
        ('texas_short.wav', 'texas_long.wav', 'prefix', 0.8728770796805216),
        ('texas_short.wav', 'texas_long.wav', 'resample_max', 0.8713947848528786),
        ('texas_long.wav', 'texas_short.wav', 'resample_max', 0.8713947848528786),
        ('texas_short.wav', 'texas_long.wav', 'resample_min', 0.6971805796646183),
        ('texas_long.wav', 'texas_short.wav', 'resample_min', 0.6971805796646183),

        ('texas_long.wav', 'texas_long.wav', 'none', 1.0),
        ('texas_long.wav', 'texas_long.flac', 'none', 1.0),
        ('texas_long.wav', 'texas_long.mp3', 'none', 0.9987497286333322),
    ])
    def test_sound_mfcc_similarity_time_align(self, file1, file2, time_align, v):
        file1 = get_testfile('assets', file1)
        file2 = get_testfile('assets', file2)
        if isinstance(v, type) and issubclass(v, BaseException):
            with pytest.raises(v):
                _ = sound_mfcc_similarity(file1, file2, time_align=time_align)
        else:
            assert sound_mfcc_similarity(file1, file2, time_align=time_align) == pytest.approx(v)

    @pytest.mark.parametrize(['file1', 'file2', 'sr_align', 'v'], [
        ('stereo_sine_wave.wav', 'stereo_sine_wave.wav', 'none', 1.0),
        ('stereo_sine_wave.wav', 'stereo_sine_wave.wav', 'min', 1.0),
        ('stereo_sine_wave.wav', 'stereo_sine_wave.wav', 'max', 1.0),
        ('stereo_sine_wave.wav', 'stereo_sine_wave.wav', 'bullshit', ValueError),

        ('stereo_sine_wave.wav', 'stereo_sine_wave_44100.wav', 'none', SoundResampleRateNotMatch),
        ('stereo_sine_wave.wav', 'stereo_sine_wave_44100.wav', 'min', 0.9999701859565737),
        ('stereo_sine_wave.wav', 'stereo_sine_wave_44100.wav', 'max', 0.9999913655328565),

        ('stereo_sine_wave.wav', 'stereo_sine_wave_3x_40_900.wav', 'none', SoundResampleRateNotMatch),
        ('stereo_sine_wave.wav', 'stereo_sine_wave_3x_40_900.wav', 'min', 0.9623946880540923),
        ('stereo_sine_wave.wav', 'stereo_sine_wave_3x_40_900.wav', 'max', 0.9517846942566877),
    ])
    def test_sound_mfcc_similarity_sr_align(self, file1, file2, sr_align, v):
        file1 = get_testfile('assets', file1)
        file2 = get_testfile('assets', file2)
        if isinstance(v, type) and issubclass(v, BaseException):
            with pytest.raises(v):
                _ = sound_mfcc_similarity(file1, file2, time_align='pad', resample_rate_align=sr_align)
        else:
            assert sound_mfcc_similarity(file1, file2, time_align='pad', resample_rate_align=sr_align) \
                   == pytest.approx(v)

    @pytest.mark.parametrize(['file1', 'file2', 'mode', 'v'], [
        ('stereo_sine_wave.wav', 'stereo_sine_wave.wav', 'flat', 1.0),
        ('stereo_sine_wave.wav', 'stereo_sine_wave.wav', 'mean', 1.0),
        ('stereo_sine_wave.wav', 'stereo_sine_wave_44100.wav', 'flat', 0.9999701859565737),
        ('stereo_sine_wave.wav', 'stereo_sine_wave_44100.wav', 'mean', 0.9999993147811599),

        ('texas_long.wav', 'texas_long.wav', 'flat', 1.0),
        ('texas_long.wav', 'texas_long.wav', 'mean', 1.0),
        ('texas_long.wav', 'texas_long_sr8000.wav', 'flat', 0.999767841519668),
        ('texas_long.wav', 'texas_long_sr16000.wav', 'flat', 0.9998774575084983),
        ('texas_long.wav', 'texas_long.wav', 'bullshit', ValueError),
    ])
    def test_sound_mfcc_similarity_pad_min(self, file1, file2, mode, v):
        file1 = get_testfile('assets', file1)
        file2 = get_testfile('assets', file2)
        if isinstance(v, type) and issubclass(v, BaseException):
            with pytest.raises(v):
                _ = sound_mfcc_similarity(file1, file2, mode=mode, time_align='pad', resample_rate_align='min')
        else:
            assert sound_mfcc_similarity(file1, file2, mode=mode, time_align='pad', resample_rate_align='min') \
                   == pytest.approx(v)
