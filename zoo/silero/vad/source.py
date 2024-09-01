import warnings
from pprint import pprint
from typing import Callable, Optional

import numpy as np
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from soundutils.data import Sound
from soundutils.utils import open_onnx_model

languages = ['ru', 'en', 'de', 'es']


class OnnxWrapper:
    def __init__(self, path):
        self.session = open_onnx_model(path)
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(0, dtype=np.float32)
        self._last_sr = 0
        self._last_batch_size = 0
        self.reset_states()
        self.sample_rates = [8000, 16000]

    def _validate_input(self, x, sampling_rate: int):
        if x.ndim == 1:
            x = x[None, ...]
        if x.ndim > 2:
            raise ValueError(f"Too many dimensions for input audio chunk {x.dim()}")

        if sampling_rate != 16000 and (sampling_rate % 16000 == 0):
            step = sampling_rate // 16000
            x = x[:, ::step]
            sampling_rate = 16000

        if sampling_rate not in self.sample_rates:
            raise ValueError(f"Supported sampling rates: {self.sample_rates} (or multiply of 16000)")
        if sampling_rate / x.shape[1] > 31.25:
            raise ValueError("Input audio chunk is too short")

        return x, sampling_rate

    def reset_states(self, batch_size=1):
        self._state = np.zeros((2, batch_size, 128), dtype=np.float32)
        self._context = np.zeros(0, dtype=np.float32)
        self._last_sr = 0
        self._last_batch_size = 0

    def __call__(self, x, sampling_rate: int):
        x, sampling_rate = self._validate_input(x, sampling_rate)
        num_samples = 512 if sampling_rate == 16000 else 256

        if x.shape[-1] != num_samples:
            raise ValueError(
                f"Provided number of samples is {x.shape[-1]} "
                f"(Supported values: 256 for 8000 sample rate, 512 for 16000)")

        batch_size = x.shape[0]
        context_size = 64 if sampling_rate == 16000 else 32

        if not self._last_batch_size:
            self.reset_states(batch_size)
        if self._last_sr and (self._last_sr != sampling_rate):
            self.reset_states(batch_size)
        if self._last_batch_size and (self._last_batch_size != batch_size):
            self.reset_states(batch_size)

        if not len(self._context):
            self._context = np.zeros((batch_size, context_size), dtype=np.float32)

        x = np.concatenate([self._context, x], axis=1)
        if sampling_rate in [8000, 16000]:
            ort_inputs = {'input': x, 'state': self._state, 'sr': np.array(sampling_rate, dtype='int64')}
            ort_outs = self.session.run(None, ort_inputs)
            out, state = ort_outs
            self._state = state
        else:
            raise ValueError()

        self._context = x[..., -context_size:]
        self._last_sr = sampling_rate
        self._last_batch_size = batch_size

        return out


def get_speech_timestamps(
        audio: np.ndarray,
        model: OnnxWrapper,
        threshold: float = 0.5,
        sampling_rate: int = 16000,
        min_speech_duration_ms: int = 250,
        max_speech_duration_s: float = float('inf'),
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 30,
        return_seconds: bool = False,
        progress_tracking_callback: Callable[[float], None] = None,
        neg_threshold: float = None,
        window_size_samples: Optional[int] = None,
):
    """
    This method is used for splitting long audios into speech chunks using silero VAD

    Parameters
    ----------
    audio: torch.Tensor, one dimensional float torch.Tensor, other types are cast to torch if possible

    model: preloaded .jit/.onnx silero VAD model

    threshold: float (default - 0.5)
        Speech threshold. Silero VAD outputs speech probabilities for each audio chunk, probabilities ABOVE this value
        are considered as SPEECH.
        It is better to tune this parameter for each dataset separately, but "lazy" 0.5 is pretty good
        for most datasets.

    sampling_rate: int (default - 16000)
        Currently silero VAD models support 8000 and 16000 (or multiply of 16000) sample rates

    min_speech_duration_ms: int (default - 250 milliseconds)
        Final speech chunks shorter min_speech_duration_ms are thrown out

    max_speech_duration_s: int (default -  inf)
        Maximum duration of speech chunks in seconds
        Chunks longer than max_speech_duration_s will be split at the timestamp of the last silence that lasts
        more than 100ms (if any), to prevent agressive cutting.
        Otherwise, they will be split aggressively just before max_speech_duration_s.

    min_silence_duration_ms: int (default - 100 milliseconds)
        In the end of each speech chunk wait for min_silence_duration_ms before separating it

    speech_pad_ms: int (default - 30 milliseconds)
        Final speech chunks are padded by speech_pad_ms each side

    return_seconds: bool (default - False)
        whether return timestamps in seconds (default - samples)

    progress_tracking_callback: Callable[[float], None] (default - None)
        callback function taking progress in percents as an argument

    neg_threshold: float (default = threshold - 0.15)
        Negative threshold (noise or exit threshold). If model's current state is SPEECH, values BELOW this value are
        considered as NON-SPEECH.

    window_size_samples: int (default - 512 samples)
        !!! DEPRECATED, DOES NOTHING !!!

    Returns
    ----------
    speeches: list of dicts
        list containing ends and beginnings of speech chunks (samples or seconds based on return_seconds)
    """
    assert isinstance(audio, np.ndarray)
    if len(audio.shape) == 2:
        if audio.shape[0] != 1:
            raise ValueError('Please use a mono-channel audio.')
        else:
            audio = audio[0]
    elif len(audio.shape) > 2:
        raise ValueError('Unrecognizable audio waveform data.')

    if sampling_rate > 16000 and (sampling_rate % 16000 == 0):
        step = sampling_rate // 16000
        sampling_rate = 16000
        audio = audio[::step]
        warnings.warn('Sampling rate is a multiply of 16000, casting to 16000 manually!')
    else:
        step = 1

    if sampling_rate not in [8000, 16000]:
        raise ValueError("Currently silero VAD models support 8000 and 16000 (or multiply of 16000) sample rates")

    window_size_samples = window_size_samples or (512 if sampling_rate == 16000 else 256)

    model.reset_states()
    min_speech_samples = sampling_rate * min_speech_duration_ms / 1000
    speech_pad_samples = sampling_rate * speech_pad_ms / 1000
    max_speech_samples = sampling_rate * max_speech_duration_s - window_size_samples - 2 * speech_pad_samples
    min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
    min_silence_samples_at_max_speech = sampling_rate * 98 / 1000

    audio_length_samples = len(audio)

    speech_probs = []
    for current_start_sample in tqdm(range(0, audio_length_samples, window_size_samples)):
        chunk = audio[current_start_sample: current_start_sample + window_size_samples]
        if len(chunk) < window_size_samples:
            # chunk = torch.nn.functional.pad(chunk, (0, int(window_size_samples - len(chunk))))
            chunk = np.pad(chunk, (0, int(window_size_samples - len(chunk))), mode='constant', constant_values=0)
        speech_prob = model(chunk, sampling_rate).item()
        speech_probs.append(speech_prob)
        # calculate progress and seng it to callback function
        progress = current_start_sample + window_size_samples
        if progress > audio_length_samples:
            progress = audio_length_samples
        progress_percent = (progress / audio_length_samples) * 100
        if progress_tracking_callback:
            progress_tracking_callback(progress_percent)

    triggered = False
    speeches = []
    current_speech = {}

    if neg_threshold is None:
        neg_threshold = threshold - 0.15
    temp_end = 0  # to save potential segment end (and tolerate some silence)
    prev_end = next_start = 0  # to save potential segment limits in case of maximum segment size reached

    for i, speech_prob in enumerate(speech_probs):
        if (speech_prob >= threshold) and temp_end:
            temp_end = 0
            if next_start < prev_end:
                next_start = window_size_samples * i

        if (speech_prob >= threshold) and not triggered:
            triggered = True
            current_speech['start'] = window_size_samples * i
            continue

        if triggered and (window_size_samples * i) - current_speech['start'] > max_speech_samples:
            if prev_end:
                current_speech['end'] = prev_end
                speeches.append(current_speech)
                current_speech = {}
                if next_start < prev_end:  # previously reached silence (< neg_thres) and is still not speech (< thres)
                    triggered = False
                else:
                    current_speech['start'] = next_start
                prev_end = next_start = temp_end = 0
            else:
                current_speech['end'] = window_size_samples * i
                speeches.append(current_speech)
                current_speech = {}
                prev_end = next_start = temp_end = 0
                triggered = False
                continue

        if (speech_prob < neg_threshold) and triggered:
            if not temp_end:
                temp_end = window_size_samples * i
            # condition to avoid cutting in very short silence
            if ((window_size_samples * i) - temp_end) > min_silence_samples_at_max_speech:
                prev_end = temp_end
            if (window_size_samples * i) - temp_end < min_silence_samples:
                continue
            else:
                current_speech['end'] = temp_end
                if (current_speech['end'] - current_speech['start']) > min_speech_samples:
                    speeches.append(current_speech)
                current_speech = {}
                prev_end = next_start = temp_end = 0
                triggered = False
                continue

    if current_speech and (audio_length_samples - current_speech['start']) > min_speech_samples:
        current_speech['end'] = audio_length_samples
        speeches.append(current_speech)

    for i, speech in enumerate(speeches):
        if i == 0:
            speech['start'] = int(max(0, speech['start'] - speech_pad_samples))
        if i != len(speeches) - 1:
            silence_duration = speeches[i + 1]['start'] - speech['end']
            if silence_duration < 2 * speech_pad_samples:
                speech['end'] += int(silence_duration // 2)
                speeches[i + 1]['start'] = int(max(0, speeches[i + 1]['start'] - silence_duration // 2))
            else:
                speech['end'] = int(min(audio_length_samples, speech['end'] + speech_pad_samples))
                speeches[i + 1]['start'] = int(max(0, speeches[i + 1]['start'] - speech_pad_samples))
        else:
            speech['end'] = int(min(audio_length_samples, speech['end'] + speech_pad_samples))

    if return_seconds:
        for speech_dict in speeches:
            speech_dict['start'] = round(speech_dict['start'] / sampling_rate, 1)
            speech_dict['end'] = round(speech_dict['end'] / sampling_rate, 1)
    elif step > 1:
        for speech_dict in speeches:
            speech_dict['start'] *= step
            speech_dict['end'] *= step

    return speeches


if __name__ == '__main__':
    m = OnnxWrapper(
        path=hf_hub_download(
            repo_id='deepghs/silero-vad-onnx',
            repo_type='model',
            filename='silero_vad.onnx',
        )
    )
    waveform, sr = Sound.open('test_m.wav').to_mono().resample(16000).to_numpy()
    # wav = torch.from_numpy(waveform.astype(np.float32))
    wav = waveform.astype(np.float32)
    speech_timestamps = get_speech_timestamps(wav, m, sampling_rate=sr, return_seconds=True)
    pprint(speech_timestamps)
