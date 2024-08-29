from typing import List

import numpy as np
import torch
from speechbrain.inference import EncoderClassifier

from soundutils.data import Sound
from test.testings import get_testfile
from .fuckonnx import _F_STFT


class LangIdModel(torch.nn.Module):
    def __init__(self, source='speechbrain/lang-id-voxlingua107-ecapa'):
        torch.nn.Module.__init__(self)
        self.mod = EncoderClassifier.from_hparams(source=source)
        self.mod.mods.compute_features.compute_STFT = \
            _F_STFT(self.mod.mods.compute_features.compute_STFT)

    def encode_batch(self, wavs, wav_lens=None):
        # Computing features and embeddings
        # print(self.mod.mods, type(self.mod.mods))
        # quit()
        feats = self.mod.mods.compute_features(wavs)
        feats = self.mod.mods.mean_var_norm(feats, wav_lens)
        embeddings = self.mod.mods.embedding_model(feats, wav_lens)
        return embeddings

    def forward(self, w):
        # emb = self.mod.encode_batch(wavs, wav_lens)
        wavs, wav_lens = w[:, 1:], w[:, 0]
        emb = self.encode_batch(wavs, wav_lens)
        out_prob = self.mod.mods.classifier(emb).squeeze(1)
        return torch.softmax(out_prob, dim=-1)


def encode(afile, sample_rate: int = 16000):
    sound = Sound.open(afile)
    sound = sound.resample(sample_rate)
    data, sr = sound.to_numpy()
    return data[0]


def batch_encode(afiles: List[str], sample_rate: int = 16000):
    wavs = [encode(afile, sample_rate=sample_rate) for afile in afiles]
    max_len = max([ix.shape[-1] for ix in wavs])
    wav_lens = np.array([ix.shape[-1] for ix in wavs]) / max_len
    wav_lens = wav_lens.astype(np.float32)

    wavs = [np.pad(wav, (0, max_len - wav.shape[-1]), mode='constant', constant_values=0) for wav in wavs]
    return np.stack(wavs), wav_lens


def make_sample_input(dtype=torch.float32):
    wavs, wav_lens = batch_encode([
        get_testfile('assets', 'surtr_long.wav'),
        get_testfile('assets', 'surtr_assist.wav'),
        get_testfile('assets', 'surtr_short.wav'),
        get_testfile('assets', 'texas_long.wav'),
        get_testfile('assets', 'texas_assist.wav'),
        get_testfile('assets', 'texas_short.wav'),
    ])
    w = np.concatenate([wav_lens[:, None], wavs], axis=-1)
    return torch.from_numpy(w).type(dtype)


if __name__ == '__main__':
    # model = LangIdModel(source='speechbrain/lang-id-commonlanguage_ecapa')
    model = LangIdModel()
    print(model)
    w = make_sample_input()
    with torch.no_grad():
        t = model(w)
        print(torch.max(t, dim=-1))
