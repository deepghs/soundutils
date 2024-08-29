import math
import numpy as np
import pytest

from soundutils.data import hertz_to_octave


@pytest.fixture
def tuning_fixture():
    return 0.0


@pytest.fixture
def bins_per_octave_fixture():
    return 12


@pytest.fixture
def freq_single_fixture():
    return 440.0


@pytest.fixture
def freq_array_fixture():
    return np.array([220.0, 440.0, 880.0])


@pytest.mark.unittest
class TestHertzToOctave:
    def test_single_frequency(self, freq_single_fixture, tuning_fixture, bins_per_octave_fixture):
        result = hertz_to_octave(freq_single_fixture, tuning_fixture, bins_per_octave_fixture)
        assert result == 4.0, "The octave for A440 with standard tuning should be 4.0"

    def test_frequency_array(self, freq_array_fixture, tuning_fixture, bins_per_octave_fixture):
        expected_results = np.array([3.0, 4.0, 5.0])
        results = hertz_to_octave(freq_array_fixture, tuning_fixture, bins_per_octave_fixture)
        np.testing.assert_array_equal(results, expected_results,
                                      "Octave values for the array do not match expected values")

    def test_with_tuning(self, freq_single_fixture):
        tuned_octave = hertz_to_octave(freq_single_fixture, tuning=0.5, bins_per_octave=12)
        assert tuned_octave == pytest.approx(3.9583333333333335), \
            "The octave for A440 with a tuning of 0.5 should be approximately 3.958"

    def test_with_different_bins_per_octave(self, freq_single_fixture):
        octave_24_bins = hertz_to_octave(freq_single_fixture, tuning=0.0, bins_per_octave=24)
        assert octave_24_bins == pytest.approx(4.0), \
            "The octave for A440 with 24 bins per octave and standard tuning should be 4.0"

    def test_zero_frequency(self):
        # with pytest.raises(ValueError):
        assert hertz_to_octave(0) == pytest.approx(-math.inf)
