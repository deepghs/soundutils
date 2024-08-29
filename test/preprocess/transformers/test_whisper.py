import numpy as np
import pytest
from transformers import WhisperFeatureExtractor as OriginWhisperFeatureExtractor

from soundutils.data import Sound
from soundutils.preprocess.transformers import WhisperFeatureExtractor
from test.testings import get_testfile


@pytest.fixture()
def whisper_fe_config():
    return {
        "chunk_length": 30,
        "feature_extractor_type": "WhisperFeatureExtractor",
        "feature_size": 80,
        "hop_length": 160,
        "n_fft": 400,
        "n_samples": 480000,
        "nb_max_frames": 3000,
        "padding_side": "right",
        "padding_value": 0.0,
        "processor_class": "WhisperProcessor",
        "return_attention_mask": False,
        "sampling_rate": 16000
    }


@pytest.fixture()
def whisper_fe_official(whisper_fe_config):
    return OriginWhisperFeatureExtractor.from_dict(whisper_fe_config)


@pytest.fixture()
def whisper_fe(whisper_fe_config):
    return WhisperFeatureExtractor(**whisper_fe_config)


@pytest.mark.unittest
class TestPreprocessTransformersWhisper:
    @pytest.mark.parametrize(['files'], [
        (['texas_long.wav'],),
        (['texas_assist.wav'],),
        (['surtr_long.wav'],),
        (['surtr_assist.wav'],),
        (['texas_long.wav', 'texas_assist.wav'],),
        (['texas_long.wav', 'texas_assist.wav', 'surtr_long.wav', 'surtr_assist.wav'],),
    ])
    def test_whisper_feature_extractor(self, files, whisper_fe_official, whisper_fe):
        datas = []
        for file in files:
            sound = Sound.open(get_testfile('assets', file))
            sound = sound.resample(16000)
            data, sr = sound.to_numpy()
            datas.append(data[0])

        expected = whisper_fe_official(datas, return_tensors='np')
        actual = whisper_fe(datas)

        assert set(expected.keys()) == {'input_features'}
        assert set(actual.keys()) == {'input_features'}
        expected = expected['input_features']
        actual = actual['input_features']
        assert np.isclose(expected, actual, atol=1e-4).all()
