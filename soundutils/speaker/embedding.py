from functools import lru_cache

import numpy as np
from huggingface_hub import hf_hub_download

from ..data import SoundTyping, Sound
from ..utils import open_onnx_model


@lru_cache()
def _open_model():
    return open_onnx_model(hf_hub_download(
        repo_id='deepghs/pyannote-embedding-onnx',
        repo_type='model',
        filename='model.onnx',
    ))


def speaker_embedding(sound: SoundTyping, resample_aligned: bool = False):
    sound = Sound.load(sound)
    sound = sound.resample(16000, aligned=resample_aligned).to_mono()
    data, sr = sound.to_numpy()

    model = _open_model()
    embeddings, = model.run(['embeddings'], {
        'waveform': data.astype(np.float32),
    })
    return embeddings[0]
