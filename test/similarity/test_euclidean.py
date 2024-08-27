import pytest

from soundutils.similarity import sound_euclidean, SoundLengthNotMatch, SoundChannelsNotMatch, SoundResampleRateNotMatch
from test.testings import get_testfile


@pytest.mark.unittest
class TestSimilarityEuclidean:
    @pytest.mark.parametrize(['file1', 'file2', 'v'], [
        ('texas_short.wav', 'texas_short.wav', 0.0),
        ('texas_short.wav', 'texas_long.wav', SoundLengthNotMatch),
        ('texas_long.wav', 'texas_long.wav', 0.0),
        ('texas_long.wav', 'texas_long.flac', 0.0),
        ('texas_long.wav', 'texas_long.mp3', 4.130920220902324),
        ('stereo_sine_wave.wav', 'stereo_sine_wave_44100.wav', SoundResampleRateNotMatch),
        ('texas_short.wav', 'stereo_sine_wave_44100.wav', SoundChannelsNotMatch),
    ])
    def test_euclidean(self, file1, file2, v):
        file1 = get_testfile('assets', file1)
        file2 = get_testfile('assets', file2)
        if isinstance(v, type) and issubclass(v, BaseException):
            with pytest.raises(v):
                _ = sound_euclidean(file1, file2)
        else:
            assert sound_euclidean(file1, file2) == pytest.approx(v)
