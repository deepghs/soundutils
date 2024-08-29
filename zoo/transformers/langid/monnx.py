import json
from pprint import pprint

import numpy as np
from imgutils.utils import open_onnx_model

from soundutils.data import Sound
from soundutils.preprocess.transformers.whisper import WhisperFeatureExtractor
from test.testings import get_testfile

with open('onnxs/preprocessor_config.json', 'r') as f:
    cf = json.load(f)
del cf['feature_extractor_type']
pprint(cf)
fr = WhisperFeatureExtractor(**cf)


def px(file):
    sound = Sound.open(file)
    sound = sound.resample(16000)
    data, sr = sound.to_numpy()
    return fr(data[0], return_tensors='np')


def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    sum_exp_x = np.sum(exp_x, axis=axis, keepdims=True)
    return exp_x / sum_exp_x


if __name__ == '__main__':
    onnx_file = 'onnxs/model.onnx'
    model = open_onnx_model(onnx_file)

    d = px(get_testfile('assets', 'surtr_assist.wav'))
    print(d)
    print(d['input_features'].dtype, d['input_features'].shape)

    logits, = model.run(['logits'], {
        'input_features': d['input_features'],
    })

    sx = softmax(logits, axis=-1)
    print(sx)
    print(np.argmax(sx, axis=-1))
