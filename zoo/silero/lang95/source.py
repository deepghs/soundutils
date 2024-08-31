import json
from pprint import pprint

import numpy as np
import onnxruntime
from huggingface_hub import hf_hub_download

from soundutils.data import Sound
from soundutils.utils import softmax
from test.testings import get_testfile

languages = ['ru', 'en', 'de', 'es']


class Validator():
    def __init__(self, path):
        self.model = onnxruntime.InferenceSession(path)

    def __call__(self, inputs: np.ndarray):
        ort_inputs = {'input': inputs}
        outs = self.model.run(None, ort_inputs)
        return outs


def read_audio(path: str,
               sampling_rate: int = 16000):
    sound = Sound.load(path).resample(sampling_rate)
    data, sr = sound.to_numpy()
    return data[0]


def get_language_and_group(wav: np.ndarray, model, lang_dict: dict, lang_group_dict: dict, top_n: int = 5):
    wav = wav.astype(np.float32)[None, ...]
    lang_logits, lang_group_logits = model(wav)

    softm = softmax(lang_logits, axis=-1)[0]
    softm_group = softmax(lang_group_logits, axis=-1)[0]

    srtd = np.argsort(softm)[::-1]
    srtd_group = np.argsort(softm_group)[::-1]

    outs = {}
    outs_group = []
    for i in range(top_n):
        prob = softm[srtd[i]].item()
        prob_group = softm_group[srtd_group[i]].item()
        outs[lang_dict[str(srtd[i].item())]] = prob
        outs_group.append((lang_group_dict[str(srtd_group[i].item())], prob_group))

    return outs, outs_group


if __name__ == '__main__':
    lang = 'jp'
    wav = read_audio(get_testfile('assets', 'langs', lang, f'{lang}_medium.wav'), sampling_rate=16000)

    repo_id = 'deepghs/silero-lang95-onnx'
    with open(hf_hub_download(repo_id=repo_id, filename='lang_dict_95.json'), 'r') as f:
        lang_dict = json.load(f)
    with open(hf_hub_download(repo_id=repo_id, filename='lang_group_dict_95.json'), 'r') as f:
        lang_group_dict = json.load(f)
    model = Validator(path=hf_hub_download(repo_id=repo_id, filename='lang_classifier_95.onnx'))

    languages, language_groups = get_language_and_group(wav, model, lang_dict, lang_group_dict, top_n=5)

    pprint(languages)
    for gp, score in language_groups:
        print(f'Language group: {gp!r} with prob {score:.3f}')
