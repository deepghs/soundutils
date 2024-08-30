# 1. visit hf.co/pyannote/embedding and accept user conditions
# 2. visit hf.co/settings/tokens to create an access token
# 3. instantiate pretrained model
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
from pyannote.audio import Model, Inference

from soundutils.data import Sound
from soundutils.speaker import speaker_embedding
from test.testings import get_testfile

model = Model.from_pretrained(
    "pyannote/embedding",
    use_auth_token=os.environ.get('HF_TOKEN')
)
model.eval()

# sound = Sound.open(get_testfile('assets', 'texas_long.wav'))
# sound = sound.resample(16000)
# data, sr = sound.to_numpy()
# # input_ = data[None]
# input_ = torch.from_numpy(data).type(torch.float32)
#
# print(input_)
# print(input_.dtype, input_.shape)
#
# with torch.no_grad():
#     output = model(input_)
#     # output = inf.conversion(output)
#     print(output)
#     print(output.dtype, output.shape)

print(speaker_embedding(get_testfile('assets', 'texas_long.wav'), resample_aligned=True))

print('')
print('--------------------------------------------------------------------')
print('')

inference = Inference(model, window="whole")

# waveform, sample_rate = inference.model.audio(get_testfile('assets', 'texas_long.wav'))
# print(waveform)
# print(waveform.dtype, waveform.shape)
# sx = Sound.from_numpy(waveform.numpy(), sample_rate)
#
# fig, axes = plt.subplots(2, 1, sharex=True, sharey=True)
# sound.plot(ax=axes[0], title='soundutils')
# sx.plot(ax=axes[1], title='infer')
# plt.show()


# quit()

embedding1 = inference(get_testfile('assets', 'texas_long.wav'))

print(embedding1)
print(embedding1.dtype, embedding1.shape)

quit()
embedding2 = inference(get_testfile('assets', 'texas_assist.wav'))
embedding3 = inference(get_testfile('assets', 'surtr_long.wav'))
embedding4 = inference(get_testfile('assets', 'surtr_assist.wav'))
# `embeddingX` is (1 x D) numpy array extracted from the file as a whole.

print(embedding1.shape, embedding1.dtype)
print(type(embedding1))

from scipy.spatial.distance import cdist

embs = np.stack([embedding1, embedding2, embedding3, embedding4])
distance = cdist(embs, embs, metric="cosine")
print(distance)
# `distance` is a `float` describing how dissimilar speakers 1 and 2 are.
