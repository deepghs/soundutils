from typing import List, Optional, Union

import numpy as np
from transformers.utils import PaddingStrategy

from .audio_utils import spectrogram, window_function, mel_filter_bank


def _np_extract_fbank_features(
        waveform_batch: np.array, n_fft: int = 400, hop_length: int = 160,
        feature_size: int = 80, sampling_rate: int = 16000) -> np.ndarray:
    mel_filters = mel_filter_bank(
        num_frequency_bins=1 + n_fft // 2,
        num_mel_filters=feature_size,
        min_frequency=0.0,
        max_frequency=8000.0,
        sampling_rate=sampling_rate,
        norm="slaney",
        mel_scale="slaney",
    )

    log_spec_batch = []
    for waveform in waveform_batch:
        log_spec = spectrogram(
            waveform,
            window_function(n_fft, "hann"),
            frame_length=n_fft,
            hop_length=hop_length,
            power=2.0,
            mel_filters=mel_filters,
            log_mel="log10",
        )
        log_spec = log_spec[:, :-1]
        log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        log_spec_batch.append(log_spec)
    log_spec_batch = np.array(log_spec_batch)
    return log_spec_batch


def zero_mean_unit_var_norm(
        input_values: List[np.ndarray], attention_mask: List[np.ndarray], padding_value: float = 0.0
) -> List[np.ndarray]:
    """
    Every array in the list is normalized to have zero mean and unit variance
    """
    if attention_mask is not None:
        attention_mask = np.array(attention_mask, np.int32)
        normed_input_values = []

        for vector, length in zip(input_values, attention_mask.sum(-1)):
            normed_slice = (vector - vector[:length].mean()) / np.sqrt(vector[:length].var() + 1e-7)
            if length < normed_slice.shape[0]:
                normed_slice[length:] = padding_value

            normed_input_values.append(normed_slice)
    else:
        normed_input_values = [(x - x.mean()) / np.sqrt(x.var() + 1e-7) for x in input_values]

    return normed_input_values


def pad(
        processed_features: np.ndarray,
        padding: Union[bool, str, PaddingStrategy] = True,
        max_length: Optional[int] = None,
        truncation: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
) -> dict:
    assert isinstance(processed_features, np.ndarray)
    # If we have a list of dicts, let's convert it in a dict of lists
    # We do this to allow using this method as a collate_fn function in PyTorch Dataloader
    # if isinstance(processed_features, (list, tuple)) and isinstance(processed_features[0], (dict, BatchFeature)):
    #     processed_features = {
    #         key: [example[key] for example in processed_features] for key in processed_features[0].keys()
    #     }
    #
    # # The model's main input name, usually `input_values`, has be passed for padding
    # if model_input_names[0] not in processed_features:
    #     raise ValueError(
    #         "You should supply an instance of `transformers.BatchFeature` or list of `transformers.BatchFeature`"
    #         f" to this method that includes {model_input_names[0]}, but you provided"
    #         f" {list(processed_features.keys())}"
    #     )

    required_input = processed_features
    return_attention_mask = (
        return_attention_mask if return_attention_mask is not None else return_attention_mask
    )

    if len(required_input) == 0:
        if return_attention_mask:
            processed_features["attention_mask"] = []
        return processed_features

    # If we have PyTorch/TF tensors or lists as inputs, we cast them as Numpy arrays
    # and rebuild them afterwards if no return_tensors is specified
    # Note that we lose the specific device the tensor may be on for PyTorch

    first_element = required_input[0]
    if isinstance(first_element, (list, tuple)):
        # first_element might be an empty list/tuple in some edge cases so we grab the first non empty element.
        index = 0
        while len(required_input[index]) == 0:
            index += 1
        if index < len(required_input):
            first_element = required_input[index][0]

    if return_tensors is None:
        if is_tf_tensor(first_element):
            return_tensors = "tf"
        elif is_torch_tensor(first_element):
            return_tensors = "pt"
        elif isinstance(first_element, (int, float, list, tuple, np.ndarray)):
            return_tensors = "np"
        else:
            raise ValueError(
                f"type of {first_element} unknown: {type(first_element)}. "
                "Should be one of a python, numpy, pytorch or tensorflow object."
            )

    for key, value in processed_features.items():
        if isinstance(value[0], (int, float)):
            processed_features[key] = to_numpy(value)
        else:
            processed_features[key] = [to_numpy(v) for v in value]

    # Convert padding_strategy in PaddingStrategy
    padding_strategy = _get_padding_strategies(padding=padding, max_length=max_length)

    required_input = processed_features[model_input_names[0]]

    batch_size = len(required_input)
    if not all(len(v) == batch_size for v in processed_features.values()):
        raise ValueError("Some items in the output dictionary have a different batch size than others.")

    truncated_inputs = []
    for i in range(batch_size):
        inputs = {k: v[i] for k, v in processed_features.items()}
        # truncation
        inputs_slice = _truncate(
            inputs,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            truncation=truncation,
        )
        truncated_inputs.append(inputs_slice)

    if padding_strategy == PaddingStrategy.LONGEST:
        # make sure that `max_length` cannot be longer than the longest truncated length
        max_length = max(len(input_slice[model_input_names[0]]) for input_slice in truncated_inputs)
        padding_strategy = PaddingStrategy.MAX_LENGTH

    batch_outputs = {}
    for i in range(batch_size):
        # padding
        outputs = _pad(
            truncated_inputs[i],
            max_length=max_length,
            padding_strategy=padding_strategy,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
        )

        for key, value in outputs.items():
            if key not in batch_outputs:
                batch_outputs[key] = []
            if value.dtype is np.dtype(np.float64):
                value = value.astype(np.float32)
            batch_outputs[key].append(value)

    return BatchFeature(batch_outputs, tensor_type=return_tensors)


def feature_extract(
        
        raw_speech: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]],
        truncation: bool = True,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        padding: Optional[str] = "max_length",
        max_length: Optional[int] = None,
        sampling_rate: int = 16000,
        do_normalize: Optional[bool] = None,
        return_token_timestamps: Optional[bool] = None,
        hop_length: int = 160,
        chunk_length: int = 30,
        padding_value: float = 0.0,
        n_fft: int = 400,
        feature_size: int = 80,
):
    n_samples = chunk_length * sampling_rate

    is_batched_numpy = isinstance(raw_speech, np.ndarray) and len(raw_speech.shape) > 1
    if is_batched_numpy and len(raw_speech.shape) > 2:
        raise ValueError(f"Only mono-channel audio is supported for input to.")
    is_batched = is_batched_numpy or (
            isinstance(raw_speech, (list, tuple)) and (isinstance(raw_speech[0], (np.ndarray, tuple, list)))
    )

    if is_batched:
        raw_speech = [np.asarray([speech], dtype=np.float32).T for speech in raw_speech]
    elif not is_batched and not isinstance(raw_speech, np.ndarray):
        raw_speech = np.asarray(raw_speech, dtype=np.float32)
    elif isinstance(raw_speech, np.ndarray) and raw_speech.dtype is np.dtype(np.float64):
        raw_speech = raw_speech.astype(np.float32)

    # always return batch
    if not is_batched:
        raw_speech = [np.asarray([raw_speech]).T]

    batched_speech = {"input_features": raw_speech}

    # convert into correct format for padding
    padded_inputs = pad(
        batched_speech,
        padding=padding,
        max_length=max_length if max_length else n_samples,
        truncation=truncation,
        pad_to_multiple_of=pad_to_multiple_of,
        return_attention_mask=return_attention_mask or do_normalize,
    )

    # zero-mean and unit-variance normalization
    if do_normalize:
        padded_inputs["input_features"] = zero_mean_unit_var_norm(
            padded_inputs["input_features"],
            attention_mask=padded_inputs["attention_mask"],
            padding_value=padding_value,
        )
        padded_inputs["input_features"] = np.stack(padded_inputs["input_features"], axis=0)

    # make sure list is in array format
    input_features = padded_inputs.get("input_features").transpose(2, 0, 1)
    input_features = _np_extract_fbank_features(
        input_features[0],
        n_fft=n_fft,
        hop_length=hop_length,
        feature_size=feature_size,
        sampling_rate=sampling_rate
    )

    if isinstance(input_features[0], List):
        padded_inputs["input_features"] = [np.asarray(feature, dtype=np.float32) for feature in input_features]
    else:
        padded_inputs["input_features"] = input_features
    if return_attention_mask:
        padded_inputs["attention_mask"] = padded_inputs["attention_mask"][:, :: hop_length]
    if return_token_timestamps is not None:
        padded_inputs["num_frames"] = [len(raw_speech_i) // hop_length for raw_speech_i in raw_speech]

    return padded_inputs
