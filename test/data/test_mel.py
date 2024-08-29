import numpy as np
import pytest

from soundutils.data import mel_to_hertz, hertz_to_mel


@pytest.mark.unittest
class TestDataMel:
    @pytest.mark.parametrize("freq, mel_scale, expected", [
        (1000, "htk", 2595.0 * np.log10(1.0 + 1000 / 700.0)),
        (np.array([440, 880]), "htk", 2595.0 * np.log10(1.0 + np.array([440, 880]) / 700.0)),
        (1000, "kaldi", 1127.0 * np.log(1.0 + 1000 / 700.0)),
        (1000, "slaney", 3.0 * 1000 / 200.0),
        (np.array([1000, 1500]), "slaney", np.array([15.0, 15.0 + np.log(1500 / 1000) * (27.0 / np.log(6.4))]))
    ])
    def test_hertz_to_mel(self, freq, mel_scale, expected):
        assert np.allclose(hertz_to_mel(freq, mel_scale), expected)

    @pytest.mark.parametrize("mels, mel_scale, expected", [
        (2595.0 * np.log10(1.0 + 1000 / 700.0), "htk", 1000),
        (1127.0 * np.log(1.0 + 1000 / 700.0), "kaldi", 1000),
        (3.0 * 1000 / 200.0, "slaney", 1000),
        (np.array([15.0, 15.0 + np.log(1500 / 1000) * (27.0 / np.log(6.4))]), "slaney", np.array([1000, 1500]))
    ])
    def test_mel_to_hertz(self, mels, mel_scale, expected):
        assert np.allclose(mel_to_hertz(mels, mel_scale), expected)

    @pytest.mark.parametrize("mel_scale, error_type", [
        ("invalid_scale", ValueError),
        ("", ValueError),
        ("HTK", ValueError)
    ])
    def test_invalid_scale(self, mel_scale, error_type):
        with pytest.raises(error_type):
            hertz_to_mel(1000, mel_scale)
        with pytest.raises(error_type):
            mel_to_hertz(1000, mel_scale)
