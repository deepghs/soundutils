import json
from functools import lru_cache
from typing import List, Dict, Tuple

import numpy as np
from hbutils.string import plural_word
from huggingface_hub import hf_hub_download

from ..data import SoundTyping, Sound
from ..preprocess.transformers import WhisperFeatureExtractor
from ..utils import open_onnx_model, softmax


@lru_cache()
def _preprocessor_config(repo_id: str):
    with open(hf_hub_download(
            repo_id=repo_id,
            repo_type='model',
            filename='preprocessor_config.json'
    ), 'r') as f:
        return json.load(f)


@lru_cache()
def _config(repo_id: str):
    with open(hf_hub_download(
            repo_id=repo_id,
            repo_type='model',
            filename='config.json'
    ), 'r') as f:
        return json.load(f)


@lru_cache()
def _open_model(repo_id: str):
    return open_onnx_model(hf_hub_download(
        repo_id=repo_id,
        repo_type='model',
        filename='model.onnx'
    ))


_DEFAULT_MODEL = 'deepghs/whisper-medium-fleurs-lang-id-onnx'


@lru_cache()
def _feature_extractor(repo_id: str):
    return WhisperFeatureExtractor(**_preprocessor_config(repo_id=repo_id))


def _audio_preprocess(sounds: List[SoundTyping], repo_id: str = _DEFAULT_MODEL, resample_rate: int = 16000):
    datas = []
    for sf in sounds:
        sound = Sound.load(sf)
        if sound.channels != 1:
            raise ValueError(f'Only 1-channel audio is supported, '
                             f'{plural_word(sound.channels, "channel")} found in {sf}.')
        sound = sound.resample(resample_rate)
        data, sr = sound.to_numpy()
        datas.append(data[0])

    fr = _feature_extractor(repo_id=repo_id)
    return fr(datas)['input_features']


def _raw_sound_langid(sound: SoundTyping, model_name: str = _DEFAULT_MODEL):
    input_ = _audio_preprocess([sound], repo_id=model_name)
    model = _open_model(repo_id=model_name)
    input_names = [input.name for input in model.get_inputs()]
    assert len(input_names) == 1, f'Non-unique input for model {model_name!r} - {input_names!r}.'
    output_names = [output.name for output in model.get_outputs()]
    assert len(output_names) == 1, f'Non-unique output for model {model_name!r} - {output_names!r}.'

    output, = model.run(output_names, {
        input_names[0]: input_
    })
    logits = output[0]
    return softmax(logits)


def whisper_langid(sound: SoundTyping, model_name: str = _DEFAULT_MODEL) -> Tuple[str, float]:
    scores = _raw_sound_langid(
        sound=sound,
        model_name=model_name,
    )
    idx = np.argmax(scores).item()
    best_label = _config(repo_id=model_name)["id2label"][str(idx)]
    best_score = scores[idx].item()
    return best_label, best_score


def whisper_langid_score(sound: SoundTyping, model_name: str = _DEFAULT_MODEL) -> Dict[str, float]:
    score = _raw_sound_langid(
        sound=sound,
        model_name=model_name,
    )
    retval = {}
    for i, v in enumerate(score.tolist()):
        label = _config(repo_id=model_name)["id2label"][str(i)]
        retval[label] = v
    return retval
