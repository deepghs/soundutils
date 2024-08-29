import matplotlib.pyplot as plt
import numpy as np
import requests
from transformers import pipeline, AudioClassificationPipeline, WhisperFeatureExtractor
from transformers.pipelines.audio_classification import ffmpeg_read
from optimum.onnxruntime import ORTModelForAudioClassification
from soundutils.data import Sound
from test.testings import get_testfile


def preprocess(inputs, sample_rate: int=16000):
    if isinstance(inputs, str):
        if inputs.startswith("http://") or inputs.startswith("https://"):
            # We need to actually check for a real protocol, otherwise it's impossible to use a local file
            # like http_huggingface_co.png
            inputs = requests.get(inputs).content
        else:
            with open(inputs, "rb") as f:
                inputs = f.read()

    if isinstance(inputs, bytes):
        inputs = ffmpeg_read(inputs, sample_rate)

    if isinstance(inputs, dict):
        # Accepting `"array"` which is the key defined in `datasets` for
        # better integration
        if not ("sampling_rate" in inputs and ("raw" in inputs or "array" in inputs)):
            raise ValueError(
                "When passing a dictionary to AudioClassificationPipeline, the dict needs to contain a "
                '"raw" key containing the numpy array representing the audio and a "sampling_rate" key, '
                "containing the sampling_rate associated with that array"
            )

        _inputs = inputs.pop("raw", None)
        if _inputs is None:
            # Remove path which will not be used from `datasets`.
            inputs.pop("path", None)
            _inputs = inputs.pop("array", None)
        in_sampling_rate = inputs.pop("sampling_rate")
        inputs = _inputs
        # if in_sampling_rate != self.feature_extractor.sampling_rate:
        if in_sampling_rate != 16000:
            import torch

            try:
                from torchaudio import functional as F
            except (ImportError, ModuleNotFoundError):
                raise ImportError(
                    "torchaudio is required to resample audio samples in AudioClassificationPipeline. "
                    "The torchaudio package can be installed through: `pip install torchaudio`."
                )

            # inputs = F.resample(
            #     torch.from_numpy(inputs), in_sampling_rate, self.feature_extractor.sampling_rate
            # ).numpy()
            inputs = F.resample(
                torch.from_numpy(inputs), in_sampling_rate, 16000
            ).numpy()

    if not isinstance(inputs, np.ndarray):
        raise TypeError("We expect a numpy ndarray as input")
    if len(inputs.shape) != 1:
        raise ValueError("We expect a single channel audio input for AudioClassificationPipeline")

    return inputs

if __name__ == "__main__":
    classifier = pipeline(
        "audio-classification",
        model="sanchit-gandhi/whisper-medium-fleurs-lang-id"
    )

    print(classifier)
    AudioClassificationPipeline
    # print(classifier.model_input_names)
    # quit()

    x = preprocess(get_testfile('assets', 'texas_long.wav'))
    print(x, type(x))
    print(x.shape, x.dtype)

    y = classifier.preprocess(get_testfile('assets', 'texas_long.wav'))
    print(y)
    print(y['input_features'].dtype, y['input_features'].shape)
    quit()
    # print(classifier.feature_extractor.sampling_rate)
    # print(classifier(get_testfile('assets', 'texas_assist.wav')))

    print(classifier.feature_extractor, type(classifier.feature_extractor))
    WhisperFeatureExtractor.from_dict()

    print(classifier.feature_extractor.model_input_names)

    # sound = Sound.open(get_testfile('assets', 'texas_long.wav'))
    # sound = sound.resample(16000)
    # print(sound.to_numpy()[0])
    # print(sound.to_numpy()[0].shape)
    #
    # sx = Sound.from_numpy(x[None, :], 16000)
    #
    # fig, axes = plt.subplots(2, 1, sharex=True, sharey=True)
    # sound.plot(ax=axes[0], title='soundutils')
    # sx.plot(ax=axes[1], title='transformers')
    # plt.show()