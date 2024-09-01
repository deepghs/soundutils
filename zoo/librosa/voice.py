import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from soundutils.data import Sound

y, sr = librosa.load('test_m.wav', sr=None, duration=120)
print(y.shape, sr)

# And compute the spectrogram magnitude and phase
S_full, phase = librosa.magphase(librosa.stft(y))

idx = slice(*librosa.time_to_frames([30, 35], sr=sr))
plt.figure(figsize=(12, 4))
librosa.display.specshow(librosa.amplitude_to_db(S_full[:, idx], ref=np.max),
                         y_axis='log', x_axis='time', sr=sr)
plt.colorbar()
plt.tight_layout()

S_filter = librosa.decompose.nn_filter(S_full,
                                       aggregate=np.median,
                                       metric='cosine',
                                       width=int(librosa.time_to_frames(2, sr=sr)))

# The output of the filter shouldn't be greater than the input
# if we assume signals are additive.  Taking the pointwise minimium
# with the input spectrum forces this.
S_filter = np.minimum(S_full, S_filter)

# We can also use a margin to reduce bleed between the vocals and instrumentation masks.
# Note: the margins need not be equal for foreground and background separation
margin_i, margin_v = 2, 10
power = 2

mask_i = librosa.util.softmask(S_filter,
                               margin_i * (S_full - S_filter),
                               power=power)

mask_v = librosa.util.softmask(S_full - S_filter,
                               margin_v * S_filter,
                               power=power)

# Once we have the masks, simply multiply them with the input spectrum
# to separate the components

S_foreground = mask_v * S_full
S_background = mask_i * S_full
print(S_background.shape, S_background.dtype)

# 将频谱图转换回时域信号
y_foreground = librosa.istft(S_foreground * phase)
y_background = librosa.istft(S_background * phase)

# 确保信号长度与原始音频相同
target_length = len(y)
y_foreground = librosa.util.fix_length(y_foreground, size=target_length)
y_background = librosa.util.fix_length(y_background, size=target_length)
print(y_foreground.dtype, y_foreground.shape)
print(y_background.dtype, y_background.shape)
print(y_foreground.shape[0] / sr)
print(sr)

s_foreground = Sound.from_numpy(y_foreground[None, ...], sr)
s_foreground.save('test_m_foreground.wav')
s_background = Sound.from_numpy(y_background[None, ...], sr)
s_background.save('test_m_background.wav')

quit()

# sphinx_gallery_thumbnail_number = 2

plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
librosa.display.specshow(librosa.amplitude_to_db(S_full[:, idx], ref=np.max),
                         y_axis='log', sr=sr)
plt.title('Full spectrum')
plt.colorbar()

plt.subplot(3, 1, 2)
librosa.display.specshow(librosa.amplitude_to_db(S_background[:, idx], ref=np.max),
                         y_axis='log', sr=sr)
plt.title('Background')
plt.colorbar()
plt.subplot(3, 1, 3)
librosa.display.specshow(librosa.amplitude_to_db(S_foreground[:, idx], ref=np.max),
                         y_axis='log', x_axis='time', sr=sr)
plt.title('Foreground')
plt.colorbar()
plt.tight_layout()
plt.show()
