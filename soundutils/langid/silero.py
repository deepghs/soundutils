import json
from functools import lru_cache
from typing import Tuple

import numpy as np
from huggingface_hub import hf_hub_download

from ..data import SoundTyping, Sound
from ..utils import open_onnx_model, softmax, vreplace

_REPO_ID = 'deepghs/silero-lang95-onnx'


@lru_cache()
def _lang_dict_95():
    with open(hf_hub_download(repo_id=_REPO_ID, filename='lang_dict_95.json'), 'r') as f:
        return json.load(f)


@lru_cache()
def _lang_group_dict_95():
    with open(hf_hub_download(repo_id=_REPO_ID, filename='lang_group_dict_95.json'), 'r') as f:
        return json.load(f)


@lru_cache()
def _open_model():
    return open_onnx_model(hf_hub_download(repo_id=_REPO_ID, filename='lang_classifier_95.onnx'))


def _raw_langid(sound: SoundTyping, top_n: int = 5):
    sound = Sound.load(sound).to_mono().resample(sample_rate=16000)
    wav, sr = sound.to_numpy()

    model = _open_model()
    lang_logits, lang_group_logits = model.run(None, {'input': wav.astype(np.float32)})
    softm = softmax(lang_logits, axis=-1)[0]
    softm_group = softmax(lang_group_logits, axis=-1)[0]
    srtd = np.argsort(softm)[::-1]
    srtd_group = np.argsort(softm_group)[::-1]

    lang_dict = _lang_dict_95()
    lang_group_dict = _lang_group_dict_95()
    scores = {}
    group_scores = []
    for i in range(top_n):
        prob = softm[srtd[i]].item()
        prob_group = softm_group[srtd_group[i]].item()
        scores[lang_dict[str(srtd[i].item())]] = prob
        group_scores.append((lang_group_dict[str(srtd_group[i].item())], prob_group))

    return scores, group_scores


def silero_langid(sound: SoundTyping) -> Tuple[str, float]:
    scores, group_scores = _raw_langid(sound=sound, top_n=1)
    lang, score = list(scores.items())[0]
    return lang, score


def silero_langid_score(sound: SoundTyping, fmt: str = 'scores', top_n: int = 5):
    scores, group_scores = _raw_langid(sound=sound, top_n=top_n)
    return vreplace(fmt, {
        'scores': scores,
        'group_scores': group_scores,
    })
