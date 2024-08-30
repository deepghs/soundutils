import numpy as np
from scipy.spatial.distance import cdist

from soundutils.data import SoundTyping
from soundutils.speaker import speaker_embedding
from test.testings import get_testfile


def encode(sound: SoundTyping):
    return speaker_embedding(sound)


if __name__ == '__main__':
    embedding1 = encode(get_testfile('assets', 'texas_long.wav'))
    embedding2 = encode(get_testfile('assets', 'texas_assist.wav'))
    embedding3 = encode(get_testfile('assets', 'surtr_long.wav'))
    embedding4 = encode(get_testfile('assets', 'surtr_assist.wav'))

    embs = np.stack([embedding1, embedding2, embedding3, embedding4])
    distance = cdist(embs, embs, metric="cosine")
    print(distance)
