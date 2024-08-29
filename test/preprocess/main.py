import json
from pprint import pprint

import numpy as np
import pytest
from imgutils.utils import open_onnx_model

from soundutils.data import Sound
from soundutils.preprocess.transformers.whisper import WhisperFeatureExtractor
from test.testings import get_testfile

with open('onnxs/preprocessor_config.json', 'r') as f:
    cf = json.load(f)
del cf['feature_extractor_type']
pprint(cf)
fr = WhisperFeatureExtractor(**cf)


def px(files):
    datas = []
    for file in files:
        sound = Sound.open(file)
        sound = sound.resample(16000)
        data, sr = sound.to_numpy()
        datas.append(data[0])
    return fr(datas, return_tensors='np')


def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    sum_exp_x = np.sum(exp_x, axis=axis, keepdims=True)
    return exp_x / sum_exp_x


@pytest.mark.unittest
class TestPreprocessMain:
    def test_case(self):
        onnx_file = 'onnxs/model.onnx'
        model = open_onnx_model(onnx_file)

        d = px([
            get_testfile('assets', 'surtr_assist.wav'),
            get_testfile('assets', 'surtr_long.wav'),
            get_testfile('assets', 'langs', 'zh', 'zh_long.wav'),
            get_testfile('assets', 'langs', 'en', 'en_long.wav'),
        ])
        # print(d)
        # print(d['input_features'].dtype, d['input_features'].shape)

        logits, = model.run(['logits'], {
            'input_features': d['input_features'],
        })

        sx = softmax(logits, axis=-1)
        print(sx)
        print(np.argmax(sx, axis=-1))
