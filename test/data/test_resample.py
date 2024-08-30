import numpy as np
import pytest
import torchaudio

from soundutils.data import waveform_resample
from test.testings import get_testfile


@pytest.mark.unittest
class TestDataResample:
    @pytest.mark.parametrize(['file'], [
        ('surtr_short.wav',),
        ('surtr_assist.wav',),
        ('surtr_long.wav',),
        ('texas_short.wav',),
        ('texas_assist.wav',),
        ('texas_long.wav',),
        ('texas_assist_sr8000.wav',),
        ('texas_assist_sr16000.wav',),
        ('texas_assist_sr44100.wav',),
    ])
    def test_waveform_resample_with_torch(self, file):
        audio_file = get_testfile('assets', file)
        waveform, sr = torchaudio.load(audio_file)

        expected_result = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=16000).numpy()
        actual_result = waveform_resample(waveform.numpy(), orig_freq=sr, new_freq=16000)

        # print(np.isclose(expected_result, actual_result, atol=1e-5).mean())
        assert np.allclose(expected_result, actual_result, atol=1e-5)
