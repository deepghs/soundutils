import os

import numpy as np
import torch
from pyannote.audio import Model
from scipy.spatial.distance import cdist

from soundutils.data import SoundTyping, Sound
from test.testings import get_testfile

model = Model.from_pretrained(
    "pyannote/embedding",
    use_auth_token=os.environ.get('HF_TOKEN')
)


def encode(sound: SoundTyping):
    sound = Sound.load(sound)
    sound = sound.resample(16000)
    data, sr = sound.to_numpy()
    input_ = torch.from_numpy(data).type(torch.float32)

    with torch.no_grad():
        output = model(input_)
        return output.numpy()[0]


if __name__ == '__main__':
    embedding1 = encode(get_testfile('assets', 'texas_long.wav'))
    print(embedding1)
    print(embedding1.dtype, embedding1.shape)

    embedding2 = encode(get_testfile('assets', 'texas_assist.wav'))
    embedding3 = encode(get_testfile('assets', 'surtr_long.wav'))
    embedding4 = encode(get_testfile('assets', 'surtr_assist.wav'))

    embs = np.stack([embedding1, embedding2, embedding3, embedding4])
    distance = cdist(embs, embs, metric="cosine")
    print(distance)
